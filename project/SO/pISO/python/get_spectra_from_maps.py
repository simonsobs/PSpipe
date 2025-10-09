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

import sys
import time
import argparse

import numpy as np
import healpy as hp

from pixell import enmap
from pspipe_utils import kspace, log, pspipe_list, transfer_function, simulation
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
parser.add_argument('--write-sim-map-start', type=int, default=-1,
                    help='If given, the last sim index (exclusive) that will '
                    'also save the simulated map to disk.')
args = parser.parse_args()

# TODO: speed up map-level operations with mnms.concurrent_op

# are we running on data or sims? if sims, are we writing any simulated maps
# to disk?
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

# get needed info from paramfile
d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

surveys = d["surveys"]
apply_kspace_filter = d["apply_kspace_filter"] # FIXME: this might not be one thing for all surveys etc.
deconvolve_pixwin = d["deconvolve_pixwin"] # FIXME: this might not be one thing for all surveys etc.
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
binned_mcm = d["binned_mcm"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

mcm_dir = d['mcm_dir']
bestfit_dir = d["best_fits_dir"]

# if data, write into spec_dir
# if sims, write spectra into simulated_spec_dir and any maps into 
# simulated_map_dir
if which == 'data':
    alms_dir = d['alms_dir']
    spec_dir = d['spec_dir']
    pspy_utils.create_directory(alms_dir)
else:
    spec_dir = d['simulated_spec_dir']
pspy_utils.create_directory(spec_dir)

if which == 'sims' and write_sim_map_start >= 0:
    map_dir = d['simulated_map_dir']
    pspy_utils.create_directory(map_dir)

# if data, we have one instance of a mapset, so we use a mixed distribution of
# mpi: over the maps in the mapset, then over the spectra. this requires all
# tasks to have all alms at the end, which in turn requires either writing the
# alms to disk (with a barrier), or TODO: an allgather and a gather.
# 
# if sims, usually we have a large number AND the maps are all correlated (so
# they can't easily be divided over tasks), so makes more sense to mpi over the
# sims.
#
# this sets up iteration over mapsets, surveys, and maps such that the code is
# the same for data and sims
n_map, sv_list, map_list = pspipe_list.get_arrays_list(d)
n_spec, sv1_list, m1_list, sv2_list, m2_list = pspipe_list.get_spectra_list(d)
so_mpi.init(True)

if which == 'data':
    # TODO: implement allgathers so first the maps can be divvied up, then 
    # the alms can be allgathered, then the spectra can be divvied up, and the 
    # computed spectra gathered and saved, without writing alms to disk.
    subtasks_alms = so_mpi.taskrange(imin=0, imax=n_map - 1)
    subtasks_spectra = so_mpi.taskrange(imin=0, imax=n_spec - 1)
    log.info(f"Running on data")
    log.info(f"Number of alms for the mpi loop: {len(subtasks_alms)}")
    log.info(f"Number of spectra for the mpi loop: {len(subtasks_spectra)}")

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
    log.info(f"Running on sims")
    log.info(f"Number of sims for the mpi loop: {len(subtasks_mapsets)}")

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
filters = {}
filter_dicts = {}
pixwins = {}
inv_pixwins = {}

# get map-level auxiliary data products
for sv in surveys:
    maps[sv] = d[f"arrays_{sv}"] # TODO: replace with maps, arrays is confusing
    nsplits[sv] = d[f"n_splits_{sv}"]

    # if data, we must iterate over (signal + noise)_k
    # if sim, we iterate over signal, noise_k separately
    if which == 'data':
        log.info(f"Running on data, {nsplits[sv]} signal+noise splits for survey {sv}")
        splits_iterator[sv] = [f'sn{k}' for k in range(nsplits[sv])]
    else:
        log.info(f"Running on sims, signal and {nsplits[sv]} noise splits ({nsplits[sv]+1} total) for survey {sv}")
        splits_iterator[sv] = ['s'] + [f'n{k}' for k in range(nsplits[sv])]

    # FIXME: this will not work for SO LF which has a different template despite
    # being the same survey
    templates[sv] = so_map.read_map(d[f"maps_{sv}_{maps[sv][0]}"][0]) # first split of first map
        
    if d[f"pixwin_{sv}"]["pix"] == "CAR" and deconvolve_pixwin:
        wy, wx = enmap.calc_window(templates[sv].data.shape,
                                   order=d[f"pixwin_{sv}"]["order"])
        wy = wy.astype(np.float32)
        wx = wx.astype(np.float32)
        pixwins[sv] = (wy[:, None] * wx[None, :])
        inv_pixwins[sv] = pixwins[sv] ** (-1)
    elif d[f"pixwin_{sv}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
        # this is a crude approximation. really, it would be something like
        # Bbl @ (w_l)^2 C_l, so it can't be easily decoupled
        pw_l = hp.pixwin(d[f"pixwin_{sv}"]["nside"])
        _, pw_b = pspy_utils.naive_binning(np.arange(len(pw_l)), pw_l, binning_file, lmax)
        pixwins[sv] = pw_b
        inv_pixwins[sv] = pw_b ** (-1)

    if d[f"pixwin_{sv}"]["pix"] == "CAR" and apply_kspace_filter:
        filter_dicts[sv] = d[f"k_filter_{sv}"]
        filters[sv] = kspace.get_kspace_filter(templates[sv],
                                               filter_dicts[sv],
                                               dtype=np.float32)

    if which == 'sims':
        f_name_cmb = bestfit_dir + "/cmb.dat"
        ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

        f_name_fg = bestfit_dir + "/fg_{}x{}.dat"
        map_list = [f"{sv}_{m}" for sv in surveys for m in maps[sv]]
        l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, map_list, lmax, spectra)
        
# get spectrum-level auxiliary data products
# TODO: replace with pspipe_operator
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

if apply_kspace_filter:
    kspace_tf_path = d["kspace_tf_path"]
    if kspace_tf_path == "analytical":
        kspace_transfer_matrix = kspace.build_analytic_kspace_filter_matrices(surveys, # FIXME: will break if any non-CAR survey
                                                                              maps,
                                                                              templates,
                                                                              filter_dicts,
                                                                              binning_file, # FIXME: assumes same binning all maps
                                                                              lmax)
    else:
        kspace_transfer_matrix = {}
        TE_corr = {}
        for spec_name in spec_name_list:
            kspace_transfer_matrix[spec_name] = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy", allow_pickle=True)
            _, TE_corr[spec_name] = so_spectra.read_ps(f"{kspace_tf_path}/TE_correction_{spec_name}.dat", spectra=spectra)

    for k, v in kspace_transfer_matrix.items():
        if np.count_nonzero(v.diagonal()) > 0:
            log.info(f'WARNING: 0 in kspace_transfer_matrix {k}')

# now we can iterate over mapsets, and maps within them
for iii in mapset_iterator:
    
    # this will get safely overwritten if which=='data'
    master_alms = {}

    for sv, m in zip(sv_iterator, map_iterator, strict=True):

        log.info(f"[Mapset {iii}] Computing alm for survey '{sv}' and map '{m}'")
        t0 = time.time()

        win_T = so_map.read_map(d[f"window_T_{sv}_{m}"])
        win_pol = so_map.read_map(d[f"window_pol_{sv}_{m}"])

        window_tuple = (win_T, win_pol)

        if win_T.pixel == "CAR": # FIXME: might not match d[f"pixwin_{sv}"]["pix"], there should only be one
            if apply_kspace_filter or deconvolve_pixwin:
                win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{m}"])
            if apply_kspace_filter:
                filter = filters[sv]
                weighted_filter = filter_dicts[sv]["weighted"]
            if deconvolve_pixwin:
                inv_pwin = inv_pixwins[sv] # if d[f"pixwin_{sv}"]["pix"] == "CAR" else None # FIXME: see above

        cal, pol_eff = d[f"cal_{sv}_{m}"], d[f"pol_eff_{sv}_{m}"]

        for k, snk in enumerate(splits_iterator[sv]):
            
            if which == 'data':
                split_idx = k
            else:
                split_idx = k - 1 # we put signal first because it is always correlated, so has biggest memory footprint
            
            if win_T.pixel == "CAR": # FIXME: might not match d[f"pixwin_{sv}"]["pix"], there should only be one

                ###################################
                # INJECT MAPS
                ###################################

                # data injection
                split = so_map.read_map(d[f"maps_{sv}_{m}"][split_idx], geometry=win_T.data.geometry)

                if d[f"src_free_maps_{sv}"] == True:
                    ps_map_name = map.replace("_srcfree.fits", ".fits")
                    if ps_map_name == map:
                        raise ValueError(f"{ps_map_name} should contain srcfree, check map names!")
                    ps_map =  so_map.read_map(ps_map_name, geometry=win_T.data.geometry)
                    ps_map.data -= split.data
                    
                    # FIXME: would be cleaner if could just make ps_map with
                    # cutoff instead of relying on the mask
                    ps_mask = so_map.read_map(d[f"ps_mask_{sv}_{m}"])
                    ps_map.data *= ps_mask.data
                    split.data += ps_map.data
                
                # sim injection, assume no bright point sources after masking
                # HERE

                if apply_kspace_filter and deconvolve_pixwin:
                    log.info(f"[Mapset {iii}] Apply kspace filter and inv pixwin on {sv} and {m}")
                    split = kspace.filter_map(split,
                                              filter,
                                              win_kspace,
                                              inv_pixwin=inv_pwin,
                                              weighted_filter=weighted_filter,
                                              use_ducc_rfft=True)
                elif apply_kspace_filter:
                    log.info(f"[Mapset {iii}] WARNING: apply kspace filter but no inv pixwin on {sv} and {m}")
                    split = kspace.filter_map(split,
                                              filter,
                                              win_kspace,
                                              inv_pixwin=None,
                                              weighted_filter=weighted_filter,
                                              use_ducc_rfft=True)

                elif deconvolve_pixwin:
                    log.info(f"[Mapset {iii}] WARNING: inv pixwin but no kspace filter on {sv} and {m}")
                    split = so_map.fourier_convolution(split,
                                                       inv_pwin,
                                                       window=win_kspace,
                                                       use_ducc_rfft=True)
                else:
                    log.info(f"[Mapset {iii}] WARNING: no kspace filter and no inv pixwin on {sv} and {m}")

                            
            elif win_T.pixel == "HEALPIX":
                ###################################
                # INJECT MAPS
                ###################################

                # data injection
                split = so_map.read_map(d[f"maps_{sv}_{m}"][split_idx])

                # sim injection

                if deconvolve_pixwin:
                    log.info(f"[Mapset {iii}] WARNING: inv pixwin but no kspace filter on {sv} and {m} (HEALPIX)")
                else:
                    log.info(f"[Mapset {iii}] WARNING: no kspace filter and no inv pixwin on {sv} and {m} (HEALPIX)")

            split = split.calibrate(cal=cal, pol_eff=pol_eff)

            if d["remove_mean"] == True:
                split = split.subtract_mean(window_tuple)

            if which == 'data':
                master_alms = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=np.complex64) # save memory, maps only single-prec anyway
                np.save(f"{alms_dir}/alms_{sv}_{m}_{split_idx}.npy", master_alms)
                master_alms = None
            else:
                master_alms[sv, m, snk] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=np.complex64) # save memory, maps only single-prec anyway

            split = None

        win_T = None
        win_pol = None
        window_tuple = None
        win_kspace = None

        log.info(f"[Mapset {iii}] Survey '{sv}' and map '{m}' alm execution time: {time.time() - t0} seconds")

    # compute the power spectra
    
    # if data, need to load alms, and so need to wait
    # for all alms to be done
    if which == 'data':
        t0 = time.time()
        log.info(f"[Rank {so_mpi.rank}]: Loading alms")
        so_mpi.barrier()

        master_alms = {}
        for sv in surveys:
            for m in maps[sv]:
                for k, snk in enumerate(splits_iterator[sv]): # k == split_idx
                    master_alms[sv, m, snk] = np.load(f"{alms_dir}/alms_{sv}_{m}_{k}.npy")
        log.info(f"[Rank {so_mpi.rank} loaded alms: {time.time() - t0} seconds")

    t0 = time.time()
    log.info(f"[Mapset {iii}] Computing spectra")

    # store all the spectra for a mapset in one file. otherwise there will be
    # too many files (O(1 million) for 1,000 ASO sims).
    ps_dict_all = {}
    for sv1, m1, sv2, m2 in zip(sv1_iterator, m1_iterator, sv2_iterator, m2_iterator, strict=True):

        # TODO: replace all below (except TE corr) with pspipe_operator
        mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{m1}x{sv2}_{m2}",
                                            spin_pairs=spin_pairs)
        
        xtra_pw1, xtra_pw2 = None, None
        if d[f"pixwin_{sv1}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
            xtra_pw1 = pixwins[sv1]
        if d[f"pixwin_{sv2}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
            xtra_pw2 = pixwins[sv2]

        # new shit here
        def get_ps(snk1, snk2):
            # compute specific cls with higher precision, save memory overall by
            # doing this per spectrum
            l, ps = so_spectra.get_spectra_pixell(master_alms[sv1, m1, snk1].astype(np.complex128),
                                                  master_alms[sv2, m2, snk2].astype(np.complex128),
                                                  spectra=spectra)
            
            lb, ps = so_spectra.bin_spectra(l,
                                            ps,
                                            binning_file,
                                            lmax,
                                            type=type,
                                            mbb_inv=mbb_inv,
                                            spectra=spectra,
                                            binned_mcm=binned_mcm)
            
            # xtra corr debiases signal-only spectra, but cross signal-noise spectra have mean 0
            # and cross noise-noise spectra are always from different splits (also mean 0)
            if apply_kspace_filter:
                if kspace_tf_path == "analytical":
                    xtra_corr = None
                elif ('s' in snk1) and ('s' in snk2):
                    xtra_corr = TE_corr[f"{sv1}_{m1}x{sv2}_{m2}"]
                else:
                    xtra_corr = None

                lb, ps = kspace.deconvolve_kspace_filter_matrix(lb,
                                                                ps,
                                                                kspace_transfer_matrix[f"{sv1}_{m1}x{sv2}_{m2}"],
                                                                spectra,
                                                                xtra_corr=xtra_corr)
                
            lb, ps = transfer_function.deconvolve_xtra_tf(lb,
                                                          ps,
                                                          spectra,
                                                          xtra_pw1=xtra_pw1,
                                                          xtra_pw2=xtra_pw2)
            
            return ps
            
        # first measure the raw per-split spectra. NOTE: this is only redundant
        # if sv1==sv2, m1==m2, and pol1==pol2.
        #
        # NOTE: this is (s, 0, 1, 2, ...) for a sim
        for snk1 in splits_iterator[sv1]:
            for snk2 in splits_iterator[sv2]:

                # ps_dict is a nested dict: (sv1, m1, snk1), (sv2, m2, snk2) -> XY -> data,
                # where XY is some pol cross
                ps_dict_all[(sv1, m1, snk1), (sv2, m2, snk2)] = get_ps(snk1, snk2)
        
        mbb_inv = None
        Bbl = None
        master_alms = None

        # then we get "derived" spectra: the mean cross, auto and noise spectrum
        # NOTE: the noise spectrum is defined as the noise in a map which is the
        # simple average over split maps. for the data, we do all of this, and 
        # save in a "explicit" format for backwards compatibility. for sims, we
        # just save crosses in a new format
        splits_auto_iterator = pspipe_list.get_splits_auto_iterator(sv1, nsplits[sv1], sv2, nsplits[sv2])
        splits_cross_iterator = pspipe_list.get_splits_auto_iterator(sv1, nsplits[sv1], sv2, nsplits[sv2])

        exists_auto = len(splits_auto_iterator) > 0
        exists_cross = len(splits_cross_iterator) > 0
        exists_noise = (len(splits_auto_iterator) > 0) and (len(splits_cross_iterator) > 0)

        if which == 'data':
            if exists_auto:
                ps_dict_auto_mean = {spec: 0 for spec in spectra}
                for spec in spectra:
                    for s1, s2 in splits_auto_iterator:
                        ps_dict_auto_mean[spec] += ps_dict_all[(sv1, m1, f'sn{s1}'), (sv2, m2, f'sn{s2}')][spec]
                    ps_dict_auto_mean[spec] /= len(splits_auto_iterator)

                spec_name_auto = f"{type}_{sv1}_{m1}x{sv2}_{m2}_auto"
                so_spectra.write_ps(spec_dir + f"/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
                    
        # need to do special wrangling of splits for mean cross of sims.
        # we want mean_ij((s + n_i)x(s + n_j)), which is:
        # mean_ij(sxs + sxn_j + n_ixs + n_ixn_j)
        if exists_cross:
            ps_dict_cross_mean = {spec: 0 for spec in spectra}
            for spec in spectra:
                for s1, s2 in splits_cross_iterator:
                    if which == 'data':
                        ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, f'sn{s1}'), (sv2, m2, f'sn{s2}')][spec]
                    else:
                        ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, 's'), (sv2, m2, f'n{s2}')][spec]
                        ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, f'n{s1}'), (sv2, m2, 's')][spec]
                        ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, f'n{s1}'), (sv2, m2, f'n{s2}')][spec]
                ps_dict_cross_mean[spec] /= len(splits_cross_iterator)

                if which == 'sims':
                    ps_dict_cross_mean[spec] += ps_dict_all[(sv1, m1, 's'), (sv2, m2, 's')][spec]

            if which == 'data':
                spec_name_cross = f"{type}_{sv1}_{m1}x{sv2}_{m2}_cross"                
                so_spectra.write_ps(spec_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)
            else:
                ps_dict_all[(sv1, m1), (sv2, m2)] = ps_dict_cross_mean

        if which == 'data':
            if exists_noise:
                ps_dict_noise_mean = {}   
                for spec in spectra:
                    # exists_noise only if sv1 == sv2
                    ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplits[sv1]
            
                spec_name_noise = f"{type}_{sv1}_{m1}x{sv2}_{m2}_noise"
                so_spectra.write_ps(spec_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)

        if which == 'data':
            spec_name_all = f"{type}_all_sn_cross_data"
        else:
            spec_name_all = f"{type}_all_sn_cross_{iii:05d}"
        np.save(f"{spec_dir}/{spec_name_all}.npy", ps_dict_all)

    ps_dict_all = None
    
    log.info(f"[Mapset {iii}] Spectra execution time: {time.time() - t0} seconds")