description = """
Gets spectra from maps or simulations. The structure of the mpi loop changes
depending on whether from maps (over maps and spectra) or simulations
(over simulations).

The simulations can optionally be written to disk (as full-res maps) if useful.

In the case of simulations, the saved spectra include all signal and noise
cross terms, since keeping them separate enables lower-variance mc covariances.

Alms are written to disk only in the case of maps, which allows for mpi to go
over the maps first and then the spectra, since these have different
combinatorics. If disk-space is an issue, this can be fixed in the future using
mpi all_gather (for the alms).

Prior to this we need to have run get_mcm_and_bbl.py. 
"""

import time
import argparse
from os.path import join as opj

import numpy as np
import healpy as hp

from pixell import enmap, enplot
from pspipe_utils import kspace, log, pspipe_list, dict_utils, misc, io
from pspy import pspy_utils, so_dict, so_map, so_mpi, sph_tools, so_mcm, so_spectra

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--start', type=int, default=-1,
                    help='The index of the first sim to run. If less than 0 '
                    '(the default), we run on the dataset specific in the ' \
                    'paramfile.')
parser.add_argument('--stop', type=int, default=-1,
                    help='The index of the last sim to run (exclusive).')
parser.add_argument('--write-sim-map-start', type=int, default=-1,
                    help='If given, the first sim index that will also save '
                    'the simulated map to disk.')
parser.add_argument('--write-sim-map-stop', type=int, default=-1,
                    help='If given, the last sim index (exclusive) that will '
                    'also save the simulated map to disk.')
parser.add_argument('--simulate-syst', action='store_true', # default False, type bool
                    help='If given, sims will sample random beam and leakage.')
parser.add_argument('--simulate-lens', action='store_true', # default False, type bool
                    help='If given, sims will lens the CMB at the map level.')
parser.add_argument('--for-kspace', action='store_true', # default False, type bool
                    help='If given, sims will contain only signal, no noise. Used to do simulations for TF computation')
args = parser.parse_args()

# TODO: speed up map-level operations with mnms.concurrent_op

# are we running on data or sims? if sims, are we writing any simulated maps
# to disk? are we doing any systematics or lensing?
which = 'data'
if args.start >= 0:
    which = 'sims'
    start = args.start 
    stop = args.stop
    assert stop > start, \
        f'{stop=} is not greater than {start=}'
    
    write_sim_map_start = args.write_sim_map_start 
    write_sim_map_stop = args.write_sim_map_stop
    if write_sim_map_start >= 0:
        assert write_sim_map_stop > write_sim_map_start, \
            f'{write_sim_map_stop=} is not greater than {write_sim_map_start=}'
        
        for iii in range(write_sim_map_start, write_sim_map_stop):
            assert iii in range(start, stop), \
                f'requested to write simulated map {iii} but sims span ' + \
                f'only {start} to {stop}'
            
    simulate_syst = args.simulate_syst
    simulate_lens = args.simulate_lens
    for_kspace = args.for_kspace

    tag = ''
    if simulate_syst:
        tag += '_syst'
    if simulate_lens:
        tag += '_lens'
    if for_kspace:
        tag += '_for_kspace'

# get needed info from paramfile
d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

surveys = d["surveys"]
apply_kspace_filter = d["apply_kspace_filter"] # FIXME: this might not be one thing for all surveys etc.
kspace_tf_path = d["kspace_tf_path"]
deconvolve_pixwin = d["deconvolve_pixwin"] # FIXME: this might not be one thing for all surveys etc.
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
binned_mcm = d["binned_mcm"]
mcm_dir = d['mcm_dir']
bestfit_dir = d["best_fits_dir"]
plot_maps = False if "plot_maps" not in d else d["plot_maps"]

if which == 'sims':
    sim_pixwin_apod_deg = d['sim_pixwin_apod_deg']
    add_white_noise_above_lmax = d['add_white_noise_above_lmax']
    white_noise_ell_taper_width = d['white_noise_ell_taper_width']
    keep_noise_models_in_memory = d['keep_noise_models_in_memory']

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

# if data, write into spec_dir
# if sims, write spectra into simulated_spec_dir and any maps into 
# simulated_map_dir
if which == 'data':
    alms_dir = d['alms_dir']
    spec_dir = d['spec_dir']
    pspy_utils.create_directory(alms_dir)
    pspy_utils.create_directory(spec_dir)
    if plot_maps:
        maps_plot_dir = d["plots_dir"] + "/maps/"
        pspy_utils.create_directory(maps_plot_dir)
    
else:
    if not for_kspace:
        spec_dir = d['sim_spec_dir']
        pspy_utils.create_directory(spec_dir)
    else:
        spec_dir = d['sim_spec_for_tf_dir']
        pspy_utils.create_directory(spec_dir)
        scenarios = ["standard", "noE", "noB"]
        filter_opt = ["filter", "nofilter"]

    if write_sim_map_start >= 0:
        sim_map_dir = d['sim_maps_dir']
        pspy_utils.create_directory(sim_map_dir)

# if data, we have one instance of a mapset, so we use a mixed distribution of
# mpi: over the maps in the mapset, then over the spectra. this requires all
# tasks to have all alms at the end, which in turn requires writing the alms to
# disk and then reading them (with a barrier). this is OK since the alms will
# be needed to estimate the noise model, and their size on disk is typically
# much less than the maps themselves due to the limited lmax
# 
# if sims, usually we have a large number AND the maps are all correlated (so
# they can't easily be divided over tasks), so makes more sense to mpi over the
# sims.
#
# this sets up iteration over mapsets, surveys, and maps such that the code is
# the same for data and sims
n_map, sv_list, map_list = pspipe_list.get_arrays_list(d)
n_spec, sv1_list, m1_list, sv2_list, m2_list = pspipe_list.get_spectra_list(d)

# convert to arrays to support advanced indexing
sv_list = np.array(sv_list)
map_list = np.array(map_list)
sv1_list = np.array(sv1_list)
m1_list = np.array(m1_list)
sv2_list = np.array(sv2_list)
m2_list = np.array(m2_list)

so_mpi.init(True)

if which == 'data':
    subtasks_alms = so_mpi.taskrange(imin=0, imax=n_map - 1)
    subtasks_spectra = so_mpi.taskrange(imin=0, imax=n_spec - 1)
    log.info(f"[Rank {so_mpi.rank}] Running on data")
    log.info(f"[Rank {so_mpi.rank}] Number of alms for the mpi loop: {len(subtasks_alms)}")
    log.info(f"[Rank {so_mpi.rank}] Number of spectra for the mpi loop: {len(subtasks_spectra)}")

    # iteration for alms
    mapset_iterator = range(1) # there is just the data
    sv_iterator = sv_list[subtasks_alms]
    map_iterator = map_list[subtasks_alms]

    # iteration for spectra
    sv1_iterator = sv1_list[subtasks_spectra]
    m1_iterator = m1_list[subtasks_spectra]
    sv2_iterator = sv2_list[subtasks_spectra]
    m2_iterator = m2_list[subtasks_spectra]
else:
    subtasks_mapsets = so_mpi.taskrange(imin=start, imax=stop - 1)
    log.info(f"[Rank {so_mpi.rank}] Running on sims")
    log.info(f"[Rank {so_mpi.rank}] Number of sims for the mpi loop: {len(subtasks_mapsets)}")

    # iteration for alms
    mapset_iterator = subtasks_mapsets
    sv_iterator = sv_list
    map_iterator = map_list

    # iteration for spectra
    sv1_iterator = sv1_list
    m1_iterator = m1_list
    sv2_iterator = sv2_list
    m2_iterator = m2_list

# prepare the templates, filters, and pixwins that might vary over surveys.
# this includes auxiliary data products first at the map-level, and then at the
# spectrum-level
maps = {}
nsplits = {}
splits_iterator = {}
templates = {}
pixwins = {}
inv_pixwins = {}
filters = {}
filter_dicts = {}

# get map-level auxiliary data products
for sv in surveys:
    maps[sv] = d[f"arrays_{sv}"] # TODO: replace with maps, arrays is confusing
    nsplits[sv] = d[f"n_splits_{sv}"]

    # if data, we must iterate over (signal + noise)_k
    # if sim, we iterate over signal, noise_k separately
    if which == 'data':
        log.info(f"[Rank {so_mpi.rank}] {nsplits[sv]} signal+noise splits for survey {sv}")
        splits_iterator[sv] = [f'sn{k}' for k in range(nsplits[sv])]
    else:
        if not for_kspace:
            log.info(f"[Rank {so_mpi.rank}] 1 signal and {nsplits[sv]} noise splits ({nsplits[sv]+1} total) for survey {sv}")
            splits_iterator[sv] = ['s'] + [f'n{k}' for k in range(nsplits[sv])]
        else:
            splits_iterator[sv] = []
            for sc in scenarios:
                log.info(f"[Rank {so_mpi.rank}] signal only sim for survey {sv} and scenario {sc}")
                splits_iterator[sv] += [f'so_{sc}']

    # FIXME: this will not work for SO LF which has a different template despite
    # being the same survey
    templates[sv] = so_map.read_map(d[f"window_kspace_{sv}_{maps[sv][0]}"])
    
    # NOTE: a map may a CAR map but have a HEALPIX pixwin, in which case we may
    # kspace filter it, but want to use a HEALPIX pixwin
    if templates[sv].pixel == "CAR":
        if d[f"pixwin_{sv}"]["pix"] == "CAR" and deconvolve_pixwin:
            wy, wx = enmap.calc_window(templates[sv].data.shape,
                                    order=d[f"pixwin_{sv}"]["order"])
            wy = wy.astype(np.float32)
            wx = wx.astype(np.float32)
            pixwins[sv] = (wy[:, None] * wx[None, :])
            inv_pixwins[sv] = pixwins[sv] ** (-1)
        else:
            pixwins[sv] = None
            inv_pixwins[sv] = None

        if apply_kspace_filter:
            filter_dicts[sv] = d[f"k_filter_{sv}"]
            filters[sv] = kspace.get_kspace_filter(templates[sv],
                                                   filter_dicts[sv],
                                                   dtype=np.float32)
        else:
            filter_dicts[sv] = None
            filters[sv] = None
    
    else:
        pixwins[sv] = None
        inv_pixwins[sv] = None    
        filter_dicts[sv] = None
        filters[sv] = None

# get spectrum-level auxiliary data products
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

if apply_kspace_filter and kspace_tf_path != "analytical" and not for_kspace:
    TE_corr = {}
    for spec_name in spec_name_list:
        _, TE_corr[spec_name] = so_spectra.read_ps(f"{kspace_tf_path}/TE_correction_{spec_name}.dat", spectra=spectra)

# instantiate on-the-fly simulation models. this involves packaging 
# power spectra and beams etc for the signal model, and the noise model
if which == 'sims':
    from pspipe_utils import simulation
    mapname_list = []
    mapnames2minfos = {}
    bl = []
    cal = []
    pol_eff = []
    if simulate_syst:
        bl_err = []
        gl = []
        gl_err = []
    else:
        bl_err = None
        gl = None
        gl_err = None

    for sv, m in zip(sv_list, map_list):
        mapname = f'{sv}_{m}'
        mapname_list.append(mapname)

        mapnames2minfos[mapname] = {
            'geometry': templates[sv].data.geometry,
            'pixwin': pixwins[sv],
            'noise_info': d[f'noise_info_{mapname}']
        }

        bl_T, bl_err_T = misc.prep_beams(d[f"beam_T_{mapname}"], norm='mono')
        bl_P, bl_err_P = misc.prep_beams(d[f"beam_pol_{mapname}"], norm='mono')
        bl.append(np.array([bl_T, bl_P]))

        cal.append(d[f"cal_{mapname}"])
        pol_eff.append(d[f"pol_eff_{mapname}"])

        if simulate_syst:
            if d[f"beam_T_{mapname}"] == d[f"beam_pol_{mapname}"]:
                bl_err.append(bl_err_T[None]) # (1, nmode, nl)
            else:
                bl_err.append(np.array([bl_err_T, bl_err_P])) # (2, nmode, nl)

            # norm by this map's pol_eff, the most recent one in pol_effs
            gl_T2E, gl_err_T2E = misc.prep_beams(d[f"leakage_beam_{mapname}_TE"], norm=pol_eff[-1])
            gl_T2B, gl_err_T2B = misc.prep_beams(d[f"leakage_beam_{mapname}_TB"], norm=pol_eff[-1])

            gl.append(np.array([gl_T2E, gl_T2B]))
            gl_err.append(np.array([gl_err_T2E, gl_err_T2B]))
    
    if simulate_lens:
        f_name_cmb = bestfit_dir + "unlensed_cmb_and_lensing.dat"
        spectra_with_lens = spectra + ['PP'] + [f'P{p}' for p in 'TEB'] + [f'{p}P' for p in 'TEB']
        ps_mat = simulation.unlensed_cmb_and_lensing_matrix_from_file(f_name_cmb, lmax + 500, spectra_with_lens)
    else:
        f_name_cmb = bestfit_dir + "cmb.dat"
        ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax + 500, spectra)

    f_name_fg = bestfit_dir + "fg_{}x{}.dat"
    _, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, mapname_list, lmax + 500, spectra)
    
    modeltags2modelinfos = dict_utils.get_noise_model_tags_to_noise_model_infos(d)

    signal_model_args = (mapnames2minfos, lmax, ps_mat, fg_mat, bl, cal, pol_eff)
    signal_model_kwargs = dict(bl_err=bl_err, gl=gl, gl_err=gl_err, pixwin_apod_deg=sim_pixwin_apod_deg)
    if not for_kspace:
        noise_model_args = (mapnames2minfos, modeltags2modelinfos)
        noise_model_kwargs = dict(add_white_noise_above_lmax=add_white_noise_above_lmax,
                                white_noise_ell_taper_width=white_noise_ell_taper_width,
                                keep_model=keep_noise_models_in_memory)
    else:
        noise_model_args = None
        noise_model_kwargs = None
    
    data_model = simulation.DataModel(signal_model_args, noise_model_args, 
                                      signal_model_kwargs=signal_model_kwargs,
                                      noise_model_kwargs=noise_model_kwargs)

# now we can iterate over mapsets, and maps within them
for iii in mapset_iterator:
    
    # this will get safely overwritten if which=='data'
    master_alms = {}

    for sv, m in zip(sv_iterator, map_iterator, strict=True):

        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] Computing alm for survey '{sv}' and map '{m}'")
        t0 = time.time()

        win_T = so_map.read_map(d[f"window_T_{sv}_{m}"])
        win_pol = so_map.read_map(d[f"window_pol_{sv}_{m}"])

        window_tuple = (win_T, win_pol)

        if win_T.pixel == "CAR":
            if apply_kspace_filter or (deconvolve_pixwin and d[f"pixwin_{sv}"]["pix"] == "CAR"):
                win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{m}"])
            if apply_kspace_filter:
                filter = filters[sv]
                weighted_filter = filter_dicts[sv]["weighted"]
            if deconvolve_pixwin and d[f"pixwin_{sv}"]["pix"] == "CAR":
                inv_pwin = inv_pixwins[sv]

        cal, pol_eff = d[f"cal_{sv}_{m}"], d[f"pol_eff_{sv}_{m}"]

        for k, snk in enumerate(splits_iterator[sv]):
            
            if which == 'data':
                split_idx = k
            else:
                split_idx = k - 1 # we put signal first because it is always correlated, so has biggest memory footprint
            
            if win_T.pixel == "CAR":

                # data injection
                if which == 'data':
                    map_fn = d[f"maps_{sv}_{m}"][split_idx]
                    split = so_map.read_map(map_fn, geometry=win_T.data.geometry)
                    if plot_maps:
                        plot = enplot.get_plots(
                            split.data, range=(1000, 300, 300), ticks=20, mask=0, downgrade=8
                        )
                        enplot.write(maps_plot_dir + f"{sv}_{m}_{split_idx}", plot)
                    
                    if d[f"src_free_maps_{sv}"] == True:
                        ps_map_fn = map_fn.replace("_srcfree.fits", ".fits")
                        if ps_map_fn == map_fn:
                            raise ValueError(f"{ps_map_fn} should contain srcfree, check map names!")
                        ps_map =  so_map.read_map(ps_map_fn, geometry=win_T.data.geometry)
                        ps_map.data -= split.data
                        
                        # TODO: would be cleaner if could just make ps_map with
                        # cutoff instead of relying on the mask
                        winname = dict_utils.get_winname_from_map(d, f'{sv}_{m}', 'T')
                        if winname != dict_utils.get_winname_from_map(d, f'{sv}_{m}', 'pol'):
                            raise NotImplementedError('Cannot currently mask srcs if T mask != pol mask')

                        for maskfn in d[f'baseline_masks_{winname}']: # TODO: could also preserve specificity of ps_mask but this is all excluded from window anyway
                            ps_map.data *= enmap.read_map(maskfn, geometry=win_T.data.geometry).astype(bool, copy=False) # cast to bool
                        split.data += ps_map.data

                        # TODO: why not kspace filter with *all* srcs subtracted
                        # and then add back just the ones we want? do we want to
                        # kspace filter the sources? the filter correction assu-
                        # mes isotropic power spectra which sources are not
                
                # sim injection, assume no bright point sources after masking
                else:
                    if snk == 's' or 'so' in snk:
                        split = data_model.get_signal_sim(f'{sv}_{m}', iii)
                    if snk == 'so_standard':
                        split_nofilt = np.copy(split)
                    if snk == 'so_noE':
                        split[1] *= 0
                        split_nofilt = np.copy(split)
                    if snk == 'so_noB':                    
                        split[2] *= 0
                        split_nofilt = np.copy(split)
                    if 'n' in snk and not for_kspace:
                        split = data_model.get_noise_sim(f'{sv}_{m}', split_idx, iii)

                    # possibly save raw map sim
                    if iii in range(write_sim_map_start, write_sim_map_stop):
                        if snk == 's':
                            split.write_map(f"{sim_map_dir}" + f"signal_sim_map{tag}_{sv}_{m}_{iii:05d}.fits")
                        else:
                            split.write_map(f"{sim_map_dir}" + f"noise_sim_map{tag}_{sv}_{m}_set{split_idx}_{iii:05d}.fits")

                if apply_kspace_filter and (deconvolve_pixwin and d[f"pixwin_{sv}"]["pix"] == "CAR"):
                    if k == 0:
                        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] Apply kspace filter and inv pixwin on {sv}, {m}")
                    if plot_maps:
                        plot = enplot.get_plots(
                            split.data * win_kspace.data, range=(1000, 300, 300), ticks=20, mask=0, downgrade=8
                        )
                        enplot.write(maps_plot_dir + f"{sv}_{m}_{split_idx}_before_filter", plot)
                    split = kspace.filter_map(split,
                                              filter,
                                              win_kspace,
                                              inv_pixwin=inv_pwin,
                                              weighted_filter=weighted_filter,
                                              use_ducc_rfft=True)
                    if plot_maps:
                        plot = enplot.get_plots(
                            split.data * win_T.data, range=(1000, 300, 300), ticks=20, mask=0, downgrade=8
                        )
                        enplot.write(maps_plot_dir + f"{sv}_{m}_{split_idx}_after_filter", plot)
                elif apply_kspace_filter:
                    if k == 0:
                        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] WARNING: apply kspace filter but no inv pixwin on {sv}, {m}")
                    if plot_maps:
                        plot = enplot.get_plots(
                            split.data * win_kspace.data, range=(1000, 300, 300), ticks=20, mask=0, downgrade=8
                        )
                        enplot.write(maps_plot_dir + f"{sv}_{m}_{split_idx}_before_filter", plot)
                    split = kspace.filter_map(split,
                                              filter,
                                              win_kspace,
                                              inv_pixwin=None,
                                              weighted_filter=weighted_filter,
                                              use_ducc_rfft=True)
                    if plot_maps:
                        if plot_maps:
                            plot = enplot.get_plots(
                                split.data * win_T.data, range=(1000, 300, 300), ticks=20, mask=0, downgrade=8
                            )
                            enplot.write(maps_plot_dir + f"{sv}_{m}_{split_idx}_after_filter", plot)

                elif deconvolve_pixwin and d[f"pixwin_{sv}"]["pix"] == "CAR":
                    if k == 0:
                        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] WARNING: inv pixwin but no kspace filter on {sv}, {m}")
                    split = so_map.fourier_convolution(split,
                                                       inv_pwin,
                                                       window=win_kspace,
                                                       use_ducc_rfft=True)
                    
                    if for_kspace:
                        split_nofilt = so_map.fourier_convolution(split_nofilt,
                                                       inv_pwin,
                                                       window=win_kspace,
                                                       use_ducc_rfft=True)
                else:
                    if k == 0:
                        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] WARNING: no kspace filter and no inv pixwin on {sv}, {m}")

            elif win_T.pixel == "HEALPIX":

                # data injection
                if which == 'data':
                    split = so_map.read_map(d[f"maps_{sv}_{m}"][split_idx])

                # sim injection
                else:
                    # TODO: implement this
                    assert False

                if deconvolve_pixwin:
                    if k == 0:
                        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] WARNING: inv pixwin but no kspace filter on {sv}, {m} (HEALPIX)")
                else:
                    if k == 0:
                        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] WARNING: no kspace filter and no inv pixwin on {sv}, {m} (HEALPIX)")

            split = split.calibrate(cal=cal, pol_eff=pol_eff)
            if for_kspace:
                split_nofilt =  split_nofilt.calibrate(cal=cal, pol_eff=pol_eff)
            
            if d["remove_mean"] == True:
                split = split.subtract_mean(window_tuple)
                if for_kspace:
                    split_nofilt =  split_nofilt.subtract_mean(window_tuple)             


            if which == 'data':
                master_alms = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=np.complex64) # save memory, maps only single-prec anyway
                np.save(f"{alms_dir}" + f"alms_{sv}_{m}_set{split_idx}.npy", master_alms)
                master_alms = None
            else:
                if not for_kspace:
                    master_alms[sv, m, snk] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=np.complex64) # save memory, maps only single-prec anyway
                else:
                    master_alms[sv, m, snk, "filter"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=np.complex64) # save memory, maps only single-prec anyway
                    master_alms[sv, m, snk, "nofilter"] = sph_tools.get_alms(split_nofilt, window_tuple, niter, lmax, dtype=np.complex64) # save memory, maps only single-prec anyway

            split = None
            if for_kspace:
                split_nofilt = None

        win_T = None
        win_pol = None
        window_tuple = None
        win_kspace = None

        log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] Survey '{sv}' and map '{m}' alm execution time: {(time.time() - t0):.3f} seconds")

    # compute the power spectra
    
    # if data, need to load alms, and so need to wait for all alms to be done to
    # load safely. NOTE: but, just need to load the alms needed for this task. 
    # this saves a lot of memory
    if which == 'data':
        t0 = time.time()
        log.info(f"[Rank {so_mpi.rank}] Loading alms")
        so_mpi.barrier()

        # FIXME: check against zip of sv1, m1 and sv2, m2
        master_alms = {}
        for sv, m in zip(sv_list, map_list):
            if (sv not in sv1_iterator) and (sv not in sv2_iterator):
                continue
            if (m not in m1_iterator) and (m not in m2_iterator):
                continue
            for k, snk in enumerate(splits_iterator[sv]): # k == split_idx, since data
                master_alms[sv, m, snk] = np.load(f"{alms_dir}" + f"alms_{sv}_{m}_set{k}.npy")
        log.info(f"[Rank {so_mpi.rank}] Loaded alms: {(time.time() - t0):.3f} seconds")

    t0 = time.time()
    log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] Computing spectra")

    # store all the spectra for a mapset in one file. otherwise there will be
    # too many files (O(1 million) for 1,000 ASO sims).
    ps_dict_all = {}
    if for_kspace:
        ps_dict_all_nofilt = {}

    for sv1, m1, sv2, m2 in zip(sv1_iterator, m1_iterator, sv2_iterator, m2_iterator, strict=True):
        spec_name = f"{sv1}_{m1}x{sv2}_{m2}"
        pseudo2datavec = np.load(opj(f'{mcm_dir}', f'pseudo2datavec_{spec_name}.npy'), allow_pickle=True).item()            


        # first measure the raw per-split spectra. NOTE: redundant computation
        # is performed when sv1==sv2 and m1==m2, but the code is cleaner
        #
        # NOTE: this is (s, 0, 1, 2, ...) for a sim
        for snk1 in splits_iterator[sv1]:
            for snk2 in splits_iterator[sv2]:

                # compute specific cls with higher precision, save memory overall by
                # doing this per spectrum. start_at_zero=False to match pspy convention
                # TODO: test if speed penalty of alm np.complex128 conversion 
                # is worth the memory saved (takes ~14s per spectrum)
                if not for_kspace:
                    _, pseudo_dict = so_spectra.get_spectra_pixell(master_alms[sv1, m1, snk1],
                                                                master_alms[sv2, m2, snk2],
                                                                spectra=spectra,
                                                                apply_pspy_cut=True,
                                                                dtype=np.float64)
                    
                    # we know this multiplication "works": pseudo2datavec and pseudo_dict
                    # have all spectra, so data_dict will too
                    data_dict = so_mcm.sparse_dict_mat_matmul_sparse_dict_vec(pseudo2datavec, pseudo_dict)

                    if apply_kspace_filter and kspace_tf_path != "analytical":
                        if ('s' in snk1) and ('s' in snk2):
                            for spec in data_dict:
                                data_dict[spec] -= TE_corr[spec_name][spec]


                    # ps_dict is a nested dict: (sv1, m1, snk1), (sv2, m2, snk2) -> XY -> data,
                    # where XY is some pol cross
                    ps_dict_all[(sv1, m1, snk1), (sv2, m2, snk2)] = data_dict

                else:
                    # do not mix the alms for the different scenarios
                    if snk1 == snk2:

                        _, pseudo_dict = so_spectra.get_spectra_pixell(master_alms[sv1, m1, snk1],
                                                                    master_alms[sv2, m2, snk2],
                                                                    spectra=spectra,
                                                                    apply_pspy_cut=True,
                                                                    dtype=np.float64)

                        data_dict = so_mcm.sparse_dict_mat_matmul_sparse_dict_vec(pseudo2datavec, pseudo_dict)


                        _, pseudo_dict_nofilt = so_spectra.get_spectra_pixell(master_alms[sv1, m1, snk1, "nofilter"],
                                                                master_alms[sv2, m2, snk2, "nofilter"],
                                                                spectra=spectra,
                                                                apply_pspy_cut=True,
                                                                dtype=np.float64)
                        
                        data_dict_nofilt = so_mcm.sparse_dict_mat_matmul_sparse_dict_vec(pseudo2datavec, pseudo_dict_nofilt)


                        ps_dict_all_nofilt[(sv1, m1, snk1), (sv2, m2, snk2)] = data_dict_nofilt
        
        pseudo2datavec = None

        # then we get "derived" spectra: the mean cross, auto and noise spectrum
        # NOTE: the noise spectrum is defined as the noise in a map which is the
        # simple average over split maps. for the data, we do all of this, and 
        # save in a "explicit" format for backwards compatibility. for sims, we
        # just save crosses in a new format
        splits_auto_iterator = pspipe_list.get_splits_auto_iterator(sv1, nsplits[sv1], sv2, nsplits[sv2])
        splits_cross_iterator = pspipe_list.get_splits_cross_iterator(sv1, nsplits[sv1], sv2, nsplits[sv2])

        exists_auto = len(splits_auto_iterator) > 0
        exists_cross = len(splits_cross_iterator) > 0
        exists_noise = (len(splits_auto_iterator) > 0) and (len(splits_cross_iterator) > 0) and not for_kspace

        if exists_auto:
            if not for_kspace:
                ps_dict_auto_mean = {spec: 0 for spec in spectra}
            else:
                ps_dict_auto_mean = {}
                ps_dict_auto_mean_nofilt = {}
                for sc in scenarios:
                    ps_dict_auto_mean[sc] = {spec: 0 for spec in spectra}
                    ps_dict_auto_mean_nofilt[sc] = {spec: 0 for spec in spectra}
            if which == 'sims' and not for_kspace:
                ps_dict_nn_mean = {spec: 0 for spec in spectra}
            for spec in spectra:
                for s1, s2 in splits_auto_iterator:
                    if which == 'data':
                        ps_dict_auto_mean[spec] += ps_dict_all[(sv1, m1, f'sn{s1}'), (sv2, m2, f'sn{s2}')][spec]
                    else:
                        if not for_kspace:
                            ps_dict_auto_mean[spec] += ps_dict_all[(sv1, m1, 's'), (sv2, m2, 's')][spec] # redundant but clear and fast
                            ps_dict_auto_mean[spec] += ps_dict_all[(sv1, m1, 's'), (sv2, m2, f'n{s2}')][spec]
                            ps_dict_auto_mean[spec] += ps_dict_all[(sv1, m1, f'n{s1}'), (sv2, m2, 's')][spec]
                            ps_dict_auto_mean[spec] += ps_dict_all[(sv1, m1, f'n{s1}'), (sv2, m2, f'n{s2}')][spec]
                            ps_dict_nn_mean[spec] += ps_dict_all[(sv1, m1, f'n{s1}'), (sv2, m2, f'n{s2}')][spec]
                        else:
                            for sc in scenarios:
                                ps_dict_auto_mean[sc][spec] += ps_dict_all[(sv1, m1, f'so_{sc}'), (sv2, m2, f'so_{sc}')][spec]
                                ps_dict_auto_mean_nofilt[sc][spec] += ps_dict_all_nofilt[(sv1, m1, f'so_{sc}'), (sv2, m2, f'so_{sc}')][spec] # redundant but clear and fast
                if not for_kspace:
                    ps_dict_auto_mean[spec] /= len(splits_auto_iterator)
                    if which == 'sims':
                        ps_dict_nn_mean[spec] /= len(splits_auto_iterator)
                else:
                    for sc in scenarios:
                        ps_dict_auto_mean[sc][spec] /= len(splits_auto_iterator)
                        ps_dict_auto_mean_nofilt[sc][spec] /= len(splits_auto_iterator)
            
            if not for_kspace:
                ps_dict_all[(sv1, m1), (sv2, m2), 'auto'] = ps_dict_auto_mean
            else:
                for sc in scenarios:
                    ps_dict_all[(sv1, m1), (sv2, m2), 'auto', sc] = ps_dict_auto_mean[sc]
                    ps_dict_all_nofilt[(sv1, m1), (sv2, m2), 'auto', sc] = ps_dict_auto_mean_nofilt[sc]
            
            if which == 'data':
                spec_name_auto = f"{type}_{sv1}_{m1}x{sv2}_{m2}_auto"
                so_spectra.write_ps(spec_dir + f"/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
                    
        if exists_cross:
            if not for_kspace:
                ps_dict_cross_mean = {spec: 0 for spec in spectra}
            else:
                ps_dict_cross_mean = {}
                ps_dict_cross_mean_nofilt = {}
                for sc in scenarios:
                    ps_dict_cross_mean[sc] = {spec: 0 for spec in spectra}
                    ps_dict_cross_mean_nofilt[sc] = {spec: 0 for spec in spectra}

            for spec in spectra:
                for s1, s2 in splits_cross_iterator:
                    if which == 'data':
                        ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, f'sn{s1}'), (sv2, m2, f'sn{s2}')][spec]
                    else:
                        if not for_kspace:
                            ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, 's'), (sv2, m2, 's')][spec] # redundant but clear and fast
                            ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, 's'), (sv2, m2, f'n{s2}')][spec]
                            ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, f'n{s1}'), (sv2, m2, 's')][spec]
                            ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, f'n{s1}'), (sv2, m2, f'n{s2}')][spec]
                        else:
                            for sc in scenarios:
                                ps_dict_cross_mean[sc][spec] += ps_dict_all[(sv1, m1, f'so_{sc}'), (sv2, m2, f'so_{sc}')][spec]
                                ps_dict_cross_mean_nofilt[sc][spec] += ps_dict_all_nofilt[(sv1, m1, f'so_{sc}'), (sv2, m2, f'so_{sc}')][spec] # redundant but clear and fast

                if not for_kspace:
                    ps_dict_cross_mean[spec] /= len(splits_cross_iterator)
                else:
                    for sc in scenarios:
                        ps_dict_cross_mean[sc][spec] /= len(splits_cross_iterator)
                        ps_dict_cross_mean_nofilt[sc][spec] /= len(splits_cross_iterator)
            
            if not for_kspace:
                ps_dict_all[(sv1, m1), (sv2, m2), 'cross'] = ps_dict_cross_mean
            else:
                for sc in scenarios:
                    ps_dict_all[(sv1, m1), (sv2, m2), 'cross', sc] = ps_dict_cross_mean[sc]
                    ps_dict_all_nofilt[(sv1, m1), (sv2, m2), 'cross', sc] = ps_dict_cross_mean_nofilt[sc]

            if which == 'data':
                spec_name_cross = f"{type}_{sv1}_{m1}x{sv2}_{m2}_cross"                
                so_spectra.write_ps(spec_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)

        if exists_noise:
            ps_dict_noise_mean = {}   
            for spec in spectra:
                # exists_noise only if sv1 == sv2
                ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplits[sv1]
                if which == 'sims':
                    ps_dict_nn_mean[spec] /= nsplits[sv1]
            
            ps_dict_all[(sv1, m1), (sv2, m2), 'noise'] = ps_dict_noise_mean
            if which == 'data':
                spec_name_noise = f"{type}_{sv1}_{m1}x{sv2}_{m2}_noise"
                so_spectra.write_ps(spec_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)
            else:
                ps_dict_all[(sv1, m1), (sv2, m2), 'nn'] = ps_dict_nn_mean

    master_alms = None

    log.info(f"[Rank {so_mpi.rank}, Mapset {iii}] Spectra computation time: {time.time() - t0} seconds")

    if which == 'data':
        spec_name_all = f"{type}_all_sn_cross_data"

        # need to gather all the dicts from the separate mpi tasks to save into
        # one file. assert there are no overlapping keys
        ps_dict_all = so_mpi.gather_set_or_dict(ps_dict_all, allgather=False,
                                                root=0, overlap_allowed=False)
        
        if so_mpi.rank == 0:
            ps_dict_all['l'] = lb
            np.save(f"{spec_dir}" + f"{spec_name_all}.npy", ps_dict_all)

    else:
        if not for_kspace:
            spec_name_all = f"{type}{tag}_all_sn_cross_{iii:05d}"
            
            # each process has separate maps in its mapset
            np.save(f"{spec_dir}" + f"{spec_name_all}.npy", ps_dict_all)
        else:
            spec_name_all = f"{type}{tag}_all_s_cross_filter_{iii:05d}"
            spec_name_all_nofilt = f"{type}{tag}_all_s_cross_nofilter_{iii:05d}"
        
            # each process has separate maps in its mapset
            io.save_hdf5(f"{spec_dir}" + f"{spec_name_all}.h5", ps_dict_all)
            io.save_hdf5(f"{spec_dir}" + f"{spec_name_all_nofilt}.h5", ps_dict_all_nofilt)


    ps_dict_all = None
    if for_kspace:
        ps_dict_all_nofilt = None
