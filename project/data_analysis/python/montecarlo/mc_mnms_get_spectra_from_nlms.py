"""
This script generate simulations of the actpol data
it generates gaussian simulations of cmb, fg and add noise based on the mnms simulations
the fg is based on fgspectra, note that the noise sim include the pixwin so we have to convolve the signal sim with it
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi
import numpy as np
import sys
import time
from pixell import curvedsky
from pspipe_utils import simulation, pspipe_list, kspace, misc, log

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
if sim_alm_dtype in ["complex64", "complex128"]: sim_alm_dtype = getattr(np, sim_alm_dtype)
else: raise ValueError(f"Unsupported sim_alm_dtype {sim_alm_dtype}")
dtype = np.float32 if sim_alm_dtype == "complex64" else np.float64

window_dir = "windows"
mcm_dir = "mcms"
spec_dir = "sim_spectra"
bestfit_dir = "best_fits"
nlms_dir = "mnms_noise_alms"

pspy_utils.create_directory(spec_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

# prepare the tempalte and the filter
arrays, templates, filters, n_splits, filter_dicts, pixwin, inv_pixwin = {}, {}, {}, {}, {}, {}, {}
spec_name_list = pspipe_list.get_spec_name_list(d, char="_")

for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    n_splits[sv] = len(d[f"maps_{sv}_{arrays[sv][0]}"])
    log.info(f"Running with {n_splits[sv]} splits for survey {sv}")
    template_name = d[f"maps_{sv}_{arrays[sv][0]}"][0]
    templates[sv] = so_map.read_map(template_name)
    pixwin[sv] = templates[sv].get_pixwin(dtype=np.float32)
    inv_pixwin[sv] = pixwin[sv] ** -1

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
        for spec_name in spec_name_list:
            kspace_transfer_matrix[spec_name] = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy")


f_name_cmb = bestfit_dir + "/cmb.dat"
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

f_name_fg = bestfit_dir + "/fg_{}x{}.dat"
array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, array_list, lmax, spectra)

# we will use mpi over the number of simulations
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    t0 = time.time()

    # generate cmb alms and foreground alms
    # cmb alms will be of shape (3, lm) 3 standing for T,E,B

    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, array_list, lmax, dtype="complex64")

    log.info(f"[Sim n° {iii}] Generate signal alms in {time.time()-t0:.2f} s")
    master_alms = {}

    for sv in surveys:

        t1 = time.time()

        signal_alms = {}
        for ar in arrays[sv]:

            signal_alms[ar] = alms_cmb + fglms[f"{sv}_{ar}"]
            l, bl = pspy_utils.read_beam_file(d[f"beam_{sv}_{ar}"])
            signal_alms[ar] = curvedsky.almxfl(signal_alms[ar], bl)
            # since the mnms noise sim include a pixwin, we convolve the signal ones
            signal = sph_tools.alm2map(signal_alms[ar], templates[sv])
            win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])
            signal = signal.convolve_with_pixwin(niter=niter, pixwin=pixwin[sv], window=win_kspace, use_ducc_rfft=True)
            signal_alms[ar] = sph_tools.map2alm(signal, niter, lmax)

        log.info(f"[Sim n° {iii}] Convolve beam and pixwin in {time.time()-t1:.2f} s")

        wafers = sorted({ar.split("_")[0] for ar in arrays[sv]})

        for k in range(n_splits[sv]):
            for ar in arrays[sv]:

                t3 = time.time()

                win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
                win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
                window_tuple = (win_T, win_pol)
                del win_T, win_pol

                log.info(f"[Sim n° {iii}] Read window in {time.time()-t3:.2f} s")

                noise_alms = np.load(f"{nlms_dir}/mnms_nlms_{sv}_{ar}_set{k}_{iii:05d}.npy")

                t4 = time.time()

                split = sph_tools.alm2map(signal_alms[ar] + noise_alms, templates[sv])

                log.info(f"[Sim n° {iii}] alm2map for split {k} and array {ar} done in {time.time()-t4:.2f} s")

                t5 = time.time()
                if (window_tuple[0].pixel == "CAR") & (apply_kspace_filter):

                        win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])
                        split = kspace.filter_map(split,
                                                  filters[sv],
                                                  win_kspace,
                                                  inv_pixwin = inv_pixwin[sv],
                                                  weighted_filter=filter_dicts[sv]["weighted"],
                                                  use_ducc_rfft=True)

                        del win_kspace

                log.info(f"[Sim n° {iii}] Filter split {k} and array {ar} in {time.time()-t5:.2f} s")

                #cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]
                #split = split.calibrate(cal=cal, pol_eff=pol_eff)

                if d["remove_mean"] == True:
                    split = split.subtract_mean(window_tuple)

                t6 = time.time()
                master_alms[sv, ar, k] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)
                log.info(f"[Sim n° {iii}] Final map2alm done in {time.time()-t6:.2f} s")

    ps_dict = {}

    t7 = time.time()
    
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

                lb, ps = kspace.deconvolve_kspace_filter_matrix(lb,
                                                                ps,
                                                                kspace_transfer_matrix[f"{sv1}_{ar1}x{sv2}_{ar2}"],
                                                                spectra)

                if write_all_spectra:
                    so_spectra.write_ps(spec_dir + "/{spec_name}_%05d.dat" % (iii), lb, ps, type, spectra=spectra)

                for count, spec in enumerate(spectra):
                    if (s1 == s2) & (sv1 == sv2):
                        if count == 0:  log.info(f"[Sim n° {iii}] auto {sv1}_{ar1} X {sv2}_{ar2} {s1}{s2}")
                        ps_dict[spec, "auto"] += [ps[spec]]
                    else:
                        if count == 0: log.info(f"[Sim n° {iii}] cross {sv1}_{ar1} X {sv2}_{ar2} {s1}{s2}")
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
        so_spectra.write_ps(spec_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)

        if sv1 == sv2:
            spec_name_auto = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_auto_{iii:05d}"
            so_spectra.write_ps(spec_dir + f"/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
            spec_name_noise = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_noise_{iii:05d}"
            so_spectra.write_ps(spec_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)

    log.info(f"[Sim n° {iii}] Spectra computation done in {time.time()-t7:.2f} s")
    log.info(f"[Sim n° {iii}] Done in {time.time()-t0:.2f} s")