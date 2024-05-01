description = """
This script generate simulations of the actpol data
it generates gaussian simulations of cmb, fg and add noise based on the mnms simulations
the fg is based on fgspectra, note that the noise sim include the pixwin so we have to deconvolve it only from the noise
"""

import argparse
import os
import time

import healpy as hp
import numpy as np
from pixell import enmap
from mnms import noise_models as nm, utils
from pspipe_utils import kspace, log, misc, pspipe_list, simulation, transfer_function
from pspy import pspy_utils, so_dict, so_map, so_mcm, so_spectra, sph_tools

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--delta-per-task', type=int, default=1,
                    help='The number of sims to compute in a given task.')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
sim_alm_dtype = d["sim_alm_dtype"]
lmax_noise_sim = d['lmax_noise_sim']
binned_mcm = d["binned_mcm"]
apply_kspace_filter = d["apply_kspace_filter"]
if sim_alm_dtype in ["complex64", "complex128"]: sim_alm_dtype = getattr(np, sim_alm_dtype)
else: raise ValueError(f"Unsupported sim_alm_dtype {sim_alm_dtype}")
dtype = np.float32 if sim_alm_dtype == "complex64" else np.float64

if d["remove_mean"] == True:
    raise ValueError('Removing the mean is an unphysical assumption')

# Aliases for arrays
arrays_alias = {
    "pa4": {"f150": "pa4a", "f220": "pa4b"},
    "pa5": {"f090": "pa5a", "f150": "pa5b"},
    "pa6": {"f090": "pa6a", "f150": "pa6b"}
}

# Load the noise models
noise_models = {
    wafer_name: nm.BaseNoiseModel.from_config("act_dr6v4",
                                              d[f"noise_sim_type_{wafer_name}"],
                                              *arrays_alias[wafer_name].values()) # NOTE: assumes insertion order 
    for sv in surveys for wafer_name in sorted({ar.split("_")[0] for ar in d[f"arrays_{sv}"]})
}

mcm_dir = d['mcms_dir']
bestfit_dir = d["best_fits_dir"]

spec_dir = d["sim_spectra_dir"]
pspy_utils.create_directory(spec_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

# prepare the tempalte and the filter
arrays, templates, filters, n_splits, filter_dicts, pixwin, inv_pixwin = {}, {}, {}, {}, {}, {}, {}
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    n_splits[sv] = len(d[f"maps_{sv}_{arrays[sv][0]}"])
    log.info(f"Running with {n_splits[sv]} splits for survey {sv}")
    template_name = d[f"maps_{sv}_{arrays[sv][0]}"][0]
    templates[sv] = so_map.read_map(template_name)

    if d[f"pixwin_{sv}"]["pix"] == "CAR":
        wy, wx = enmap.calc_window(templates[sv].data.shape,
                                   order=d[f"pixwin_{sv}"]["order"])
        pixwin[sv] = (wy[:, None] * wx[None, :])
        inv_pixwin[sv] = pixwin[sv] ** (-1)
    elif d[f"pixwin_{sv}"]["pix"] == "HEALPIX":
        pw_l = hp.pixwin(d[f"pixwin_{sv}"]["nside"])
        pixwin[sv] = pw_l
        inv_pixwin[sv] = pw_l ** (-1)

    if apply_kspace_filter:
        filter_dicts[sv] = d[f"k_filter_{sv}"]
        filters[sv] = kspace.get_kspace_filter(templates[sv], filter_dicts[sv], dtype=np.float32)

if apply_kspace_filter:
    kspace_tf_path = d["kspace_tf_path"]
    if kspace_tf_path == "analytical":
        kspace_transfer_matrix = kspace.build_analytic_kspace_filter_matrices(surveys,
                                                                              arrays,
                                                                              templates,
                                                                              filter_dicts,
                                                                              binning_file,
                                                                              lmax)
    else:
        kspace_transfer_matrix = {}
        TE_corr = {}
        for spec_name in spec_name_list:
            kspace_transfer_matrix[spec_name] = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy")
            _, TE_corr[spec_name] = so_spectra.read_ps(f"{kspace_tf_path}/TE_correction_{spec_name}.dat", spectra=spectra)


f_name_cmb = bestfit_dir + "/cmb.dat"
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

f_name_fg = bestfit_dir + "/fg_{}x{}.dat"
array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, array_list, lmax, spectra)

# get the sim indexes by hooking into SLURM (NOTE: in general,
# this is equivalent to embarassingly parallel applications of
# MPI where no messages are actually being passed)
delta_per_task = args.delta_per_task

job_array_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
njob_array_idxs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
job_task_idx = int(os.environ.get('SLURM_PROCID', 0))
njob_task_idxs = int(os.environ.get('SLURM_NPROCS', 1))

start = (njob_task_idxs * job_array_idx + job_task_idx) * delta_per_task
stop = (njob_task_idxs * job_array_idx + job_task_idx + 1) * delta_per_task

for iii in range(start, stop):
    t0 = time.time()

    # generate cmb alms and foreground alms
    # cmb alms will be of shape (3, lm) 3 standing for T,E,B

    # Set seed if needed
    if d["seed_sims"]:
        seed_cmb = [1, iii]
        seed_fg = [2, iii]
    else:
        seed_cmb = None
        seed_fg = None
    alms_cmb = utils.rand_alm(ps_mat, lmax=lmax, seed=seed_cmb, dtype="complex64", m_major=False) # m_major False so faster
    fglms = simulation.generate_fg_alms(fg_mat, array_list, lmax, dtype="complex64", method='mnms', seed=seed_fg, m_major=False)

    log.info(f"[Sim n° {iii}] Generate signal alms in {time.time()-t0:.2f} s")
    master_alms = {}

    for sv in surveys:

        # Get the windows, inv_pixwins, and signal for each array, then the noise for each split in an array
        for ar in arrays[sv]:
            wafer, freq = ar.split("_")

            # Get the windows and inv_pixwin
            t1 = time.time()

            win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
            if d[f"window_pol_{sv}_{ar}"] != d[f"window_T_{sv}_{ar}"]:
                win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
            else:
                win_pol = win_T  # reduce one I/O
            window_tuple = (win_T, win_pol)

            if (window_tuple[0].pixel == "CAR") & (apply_kspace_filter):
                win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])
                inv_pwin = inv_pixwin[sv] if d[f"pixwin_{sv}"]["pix"] == "CAR" else None

            cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]

            log.info(f"[Sim n° {iii}] Read window in {time.time()-t1:.2f} s")

            # Get the signal
            t2 = time.time()

            signal_alm = alms_cmb + fglms[f"{sv}_{ar}"]
            l, bl = misc.read_beams(d[f"beam_T_{sv}_{ar}"], d[f"beam_pol_{sv}_{ar}"])
            signal_alm = misc.apply_beams(signal_alm, bl)

            # only deconvolve pixwin from noise
            signal_map = sph_tools.alm2map(signal_alm, templates[sv])
            signal_alm = None

            log.info(f"[Sim n° {iii}] Convolve signal with beam for array {ar} in {time.time()-t2:.2f} s")

            t3 = time.time()
            if (window_tuple[0].pixel == "CAR") & (apply_kspace_filter):
                log.info(f'[Sim n° {iii}] Filtering signal map for array {ar}')
                signal_map = kspace.filter_map(signal_map,
                                               filters[sv],
                                               win_kspace,
                                               inv_pixwin=None, # only deconvolve pixwin from noise
                                               weighted_filter=filter_dicts[sv]["weighted"],
                                               use_ducc_rfft=True)
            log.info(f"[Sim n° {iii}] Filter signal map for array {ar} done in {time.time()-t3:.2f} s")

            t4 = time.time()
            master_alms[sv, ar, 'signal'] = sph_tools.get_alms(signal_map, window_tuple, niter, lmax, dtype=sim_alm_dtype)
            log.info(f"[Sim n° {iii}] signal map2alm for array {ar} done in {time.time()-t4:.2f} s")

            for k in range(n_splits[sv]):

                # Get the noise
                t5 = time.time()

                qid = arrays_alias[wafer][freq]
                i = 'ab'.index(qid[-1]) # FIXME: assumes form, model order of qids
                
                noise_sim_fn = noise_models[wafer].get_sim_fn(split_num=k,
                                                              sim_num=iii,
                                                              lmax=lmax_noise_sim,
                                                              alm=False,
                                                              to_write=False)
                noise_map = enmap.read_map(noise_sim_fn, sel=np.s_[i, 0])

                # calibrate noise
                noise_map[0] *= cal
                noise_map[1:3] *= cal/pol_eff

                # resample noise map onto the template
                noise_map = utils.fourier_resample(noise_map, shape=templates[sv].data.shape, wcs=templates[sv].data.wcs)
                noise_map = so_map.from_enmap(noise_map)

                log.info(f"[Sim n° {iii}] Noise map for array {ar} and split {k} done in {time.time()-t5:.2f} s")

                # deconvolve pixwin from noise
                t6 = time.time()
                if (window_tuple[0].pixel == "CAR") & (apply_kspace_filter):
                    log.info(f"[Sim n° {iii}] Filtering noise map for array {ar} and split {k}")
                    noise_map = kspace.filter_map(noise_map,
                                                  filters[sv],
                                                  win_kspace,
                                                  inv_pixwin=inv_pwin,
                                                  weighted_filter=filter_dicts[sv]["weighted"],
                                                  use_ducc_rfft=True)

                log.info(f"[Sim n° {iii}] Filter noise map for array {ar} and split {k} done in {time.time()-t6:.2f} s")

                t7 = time.time()
                master_alms[sv, ar, f'noise_{k}'] = sph_tools.get_alms(noise_map, window_tuple, niter, lmax, dtype=sim_alm_dtype)
                log.info(f"[Sim n° {iii}] noise map2alm for array {ar} and split {k} done in {time.time()-t7:.2f} s")

                # FIXME: make optional?
                # t7_2 = time.time()
                # master_alms[sv, ar, f'signal_{k}'] = sph_tools.get_alms(so_map.from_enmap(signal_map.data + noise_map.data), window_tuple, niter, lmax, dtype=sim_alm_dtype)
                # noise_map = None
                # log.info(f"[Sim n° {iii}] signal + noise map2alm for array {ar} and split {k} done in {time.time()-t7_2:.2f} s")

            win_T = None
            win_pol = None
            window_tuple = None
            win_kspace = None
            signal_map = None

    ps_dict = {}

    t8 = time.time()

    n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)

    for i_spec in range(n_spec):
        sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]

        # would be nice to load these all into memory once, but they are large
        # TODO: apply BBl to m_inv to make them small
        mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}", spin_pairs=spin_pairs)

        # under this canonization, we need to get SS, SN, NS, and NN spectra

        def get_ps(snk1, snk2):
            l, ps = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, snk1],
                                                  master_alms[sv2, ar2, snk2],
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
            if kspace_tf_path == "analytical":
                xtra_corr = None
            elif 'signal' in snk1 and 'signal' in snk2:
                xtra_corr = TE_corr[f"{sv1}_{ar1}x{sv2}_{ar2}"]
            else:
                xtra_corr = None

            lb, ps = kspace.deconvolve_kspace_filter_matrix(lb,
                                                            ps,
                                                            kspace_transfer_matrix[f"{sv1}_{ar1}x{sv2}_{ar2}"],
                                                            spectra,
                                                            xtra_corr=xtra_corr)

            # deconvolve pixwin from noise
            if d[f"pixwin_{sv1}"]["pix"] == "HEALPIX" and snk1 != 'signal':
                assert False # FIXME
                _, xtra_pw1 = pspy_utils.naive_binning(np.arange(len(pixwin[sv1])), pixwin[sv1], binning_file, lmax)
            else:
                xtra_pw1 = None
            if d[f"pixwin_{sv2}"]["pix"] == "HEALPIX" and snk2 != 'signal':
                assert False # FIXME
                _, xtra_pw2 = pspy_utils.naive_binning(np.arange(len(pixwin[sv2])), pixwin[sv2], binning_file, lmax)
            else:
                xtra_pw2 = None
            lb, ps = transfer_function.deconvolve_xtra_tf(lb,
                                                          ps,
                                                          spectra,
                                                          xtra_pw1=xtra_pw1,
                                                          xtra_pw2=xtra_pw2)
            return ps
        
        # get signal x signal spectra
        ps_dict[(sv1, ar1, 'signal'), (sv2, ar2, 'signal')] = get_ps('signal', 'signal')
        
        # get signal x noise spectra
        for s2 in range(n_splits[sv2]):
            ps_dict[(sv1, ar1, 'signal'), (sv2, ar2, f'noise_{s2}')] = get_ps('signal', f'noise_{s2}')

        # get noise x signal spectra
        for s1 in range(n_splits[sv1]):
            ps_dict[(sv1, ar1, f'noise_{s1}'), (sv2, ar2, 'signal')] = get_ps(f'noise_{s1}', 'signal')

        # get noise x noise spectra and signal_noise x signal_noise spectra
        for s1 in range(n_splits[sv1]):
            for s2 in range(n_splits[sv2]):
                ps_dict[(sv1, ar1, f'noise_{s1}'), (sv2, ar2, f'noise_{s2}')] = get_ps(f'noise_{s1}', f'noise_{s2}')
                # ps_dict[(sv1, ar1, f'signal_{s1}'), (sv2, ar2, f'signal_{s2}')] = get_ps(f'signal_{s1}', f'signal_{s2}') # FIXME: make optional?

        mbb_inv = None
        Bbl = None

    master_alms = None
    
    spec_name_cross = f"{type}_all_sn_cross_{iii:05d}"
    np.save(f"{spec_dir}/{spec_name_cross}.npy", ps_dict)

    log.info(f"[Sim n° {iii}] Spectra computation done in {time.time()-t8:.2f} s")
    log.info(f"[Sim n° {iii}] Done in {time.time()-t0:.2f} s")
