"""
This script generate simplistic simulations of the actpol data
it generates gaussian simulations of cmb, fg and noise
the fg is based on fgspectra, and the noise is based on the 1d noise power spectra measured on the data
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi, so_map_preprocessing
import numpy as np
import sys
import time
from pixell import curvedsky, powspec
from pspipe_utils import simulation, pspipe_list, best_fits, kspace, misc, log



d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
sim_alm_dtype = d["sim_alm_dtype"]
binned_mcm = d["binned_mcm"]
apply_kspace_filter = d["apply_kspace_filter"]

if sim_alm_dtype == "complex64": sim_alm_dtype = np.complex64
elif sim_alm_dtype == "complex128": sim_alm_dtype = np.complex128

mcm_dir = "mcms"
sim_spec_dir = "sim_spectra"
bestfit_dir = "best_fits"
noise_model_dir = "noise_model"

pspy_utils.create_directory(sim_spec_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

# prepare the tempalte and the filter
arrays, templates, filters, n_splits, filter_dicts = {}, {}, {}, {}, {}
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

log.info(f"build template and filter")
for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    template_name = d[f"window_T_{sv}_{arrays[sv][0]}"]
    n_splits[sv] = d[f"n_splits_{sv}"]
    log.info(f"Running with {n_splits[sv]} splits for survey {sv}")
    templates[sv] = so_map.read_map(template_name)
    if templates[sv].pixel == "CAR":
        shape, wcs = templates[sv].data.geometry
        if sim_alm_dtype == np.complex64:
            templates[sv] = so_map.car_template_from_shape_wcs(3, shape, wcs, dtype=np.float32)
        elif sim_alm_dtype == np.complex128:
            templates[sv] = so_map.car_template_from_shape_wcs(3, shape, wcs, dtype=np.float64)

    elif templates[sv].pixel == "HEALPIX":
        nside = templates[sv].nside
        templates[sv] = so_map.healpix_template(3, nside)

    if apply_kspace_filter & (templates[sv].pixel == "CAR"):
        filter_dicts[sv] = d[f"k_filter_{sv}"]
        filters[sv] = kspace.get_kspace_filter(templates[sv], filter_dicts[sv], dtype=np.float32)

log.info(f"build kspace_transfer_matrix")

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
            kspace_transfer_matrix[spec_name] = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy", allow_pickle=True)
            _, TE_corr[spec_name] = so_spectra.read_ps(f"{kspace_tf_path}/TE_correction_{spec_name}.dat", spectra=spectra)


f_name_cmb = bestfit_dir + "/cmb.dat"
f_name_noise = noise_model_dir + "/mean_{}x{}_{}_noise.dat"
f_name_fg = bestfit_dir + "/fg_{}x{}.dat"

ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)
array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, array_list, lmax, spectra)

noise_mat = {}
for sv in surveys:
    l, noise_mat[sv] = simulation.noise_matrix_from_files(f_name_noise,
                                                          sv,
                                                          arrays[sv],
                                                          lmax,
                                                          n_splits[sv],
                                                          spectra)


# we will use mpi over the number of simulations
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")
    t0 = time.time()

    # generate cmb alms and foreground alms
    # cmb alms will be of shape (3, lm) 3 standing for T,E,B

    # Set seed if needed
    if d["seed_sims"]:
        np.random.seed(iii)
    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, array_list, lmax)

    master_alms = {}

    for sv in surveys:

        t1 = time.time()

        signal_alms = {}
        for ar in arrays[sv]:
            signal_alms[ar] = alms_cmb + fglms[f"{sv}_{ar}"]
            l, bl = misc.read_beams(d[f"beam_T_{sv}_{ar}"], d[f"beam_pol_{sv}_{ar}"])
            signal_alms[ar] = misc.apply_beams(signal_alms[ar], bl)


        log.info(f"[{iii}]  Generate signal sim in {time.time() - t1:.02f} s")
        for k in range(n_splits[sv]):

            noise_alms = simulation.generate_noise_alms(noise_mat[sv], arrays[sv], lmax)
            for ar in arrays[sv]:

                t1 = time.time()

                win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
                win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
                window_tuple = (win_T, win_pol)

                log.info(f"[{iii}]  [split {k}] Reading window in {time.time()-t1:.02f} s")

                t1 = time.time()
                split = sph_tools.alm2map(signal_alms[ar] + noise_alms[ar], templates[sv])
                log.info(f"[{iii}]  [split {k}] alm2map in {time.time()-t1:.02f} s")

                t1 = time.time()
                if sv in filters:
                    
                    win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])
                    split = kspace.filter_map(split,
                                              filters[sv],
                                              win_kspace,
                                              weighted_filter=filter_dicts[sv]["weighted"],
                                              use_ducc_rfft=True)

                    del win_kspace
                    
                log.info(f"[{iii}]  [split {k}] Filtering in {time.time()-t1:.02f} s")

                if d["remove_mean"] == True:
                    split = split.subtract_mean(window_tuple)

                t1 = time.time()
                master_alms[sv, ar, k] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)
                log.info(f"[{iii}]  [split {k}] map2alm in {time.time()-t1:.02f} s")

    ps_dict = {}

    t1 = time.time()

    n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
    
    for i_spec in range(n_spec):
    
        sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]

        for spec in spectra:
            ps_dict[spec, "auto"] = []
            ps_dict[spec, "cross"] = []

        mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}",
                                            spin_pairs=spin_pairs)

        for s1 in range(n_splits[sv1]):
            for s2 in range(n_splits[sv2]):
                if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue


                l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, s1],
                                                             master_alms[sv2, ar2, s2],
                                                             spectra=spectra)

                spec_name=f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_{s1}{s2}"

                lb, ps = so_spectra.bin_spectra(l,
                                                ps_master,
                                                binning_file,
                                                lmax,
                                                type=type,
                                                mbb_inv=mbb_inv,
                                                spectra=spectra,
                                                binned_mcm=binned_mcm)

                if (sv1 in filters) & (sv2 in filters):
                
                    if kspace_tf_path == "analytical":
                        xtra_corr = None
                    else:
                        xtra_corr = TE_corr[f"{sv1}_{ar1}x{sv2}_{ar2}"]

                    lb, ps = kspace.deconvolve_kspace_filter_matrix(lb,
                                                                    ps,
                                                                    kspace_transfer_matrix[f"{sv1}_{ar1}x{sv2}_{ar2}"],
                                                                    spectra,
                                                                    xtra_corr=xtra_corr)

                if write_all_spectra:
                    so_spectra.write_ps(sim_spec_dir + f"/{spec_name}_{iii:05d}.dat", lb, ps, type, spectra=spectra)

                for count, spec in enumerate(spectra):
                    if (s1 == s2) & (sv1 == sv2):
                        ps_dict[spec, "auto"] += [ps[spec]]
                    else:
                        ps_dict[spec, "cross"] += [ps[spec]]

        ps_dict_auto_mean = {}
        ps_dict_cross_mean = {}
        ps_dict_noise_mean = {}

        for spec in spectra:
            ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)

            if ar1 == ar2 and sv1 == sv2:
                # Average TE / ET so that for same array same season TE = ET
                ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.

            if sv1 == sv2:
                ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / n_splits[sv1]


        spec_name_cross = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_cross_{iii:05d}"
        so_spectra.write_ps(sim_spec_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)

        if sv1 == sv2:
        
            spec_name_auto = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_auto_{iii:05d}"
            so_spectra.write_ps(sim_spec_dir + f"/{spec_name_auto}.dat" , lb, ps_dict_auto_mean, type, spectra=spectra)
            
            spec_name_noise = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_noise_{iii:05d}"
            so_spectra.write_ps(sim_spec_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)

    log.info(f"[{iii}]  Spectra computation in {time.time()-t1:.02f} s")
    log.info(f"[{iii}]  Simulation n° {iii:05d} done in {time.time()-t0:.02f} s")
