"""
This script generate simulations of the actpol data
it generates gaussian simulations of cmb, fg and add noise based on the mnms simulations
the fg is based on fgspectra, note that the noise sim include the pixwin so we have to convolve the signal sim with it
"""

import sys
import time

import healpy as hp
import numpy as np
from pixell import curvedsky, enmap
from pspipe_utils import kspace, log, misc, pspipe_list, simulation, transfer_function
from pspy import pspy_utils, so_dict, so_map, so_mcm, so_mpi, so_spectra, sph_tools

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


mcm_dir = "mcms"
sim_spec_dir = "sim_spectra"
bestfit_dir = "best_fits"
nlms_dir = "noise_alms"

pspy_utils.create_directory(sim_spec_dir)

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
    try:
        templates[sv] = so_map.read_map(template_name, fields_healpix=(0, 1, 2))
    except:
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

# we will use mpi over the number of simulations
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    t0 = time.time()

    # generate cmb alms and foreground alms
    # cmb alms will be of shape (3, lm) 3 standing for T,E,B

    # Set seed if needed
    if d["seed_sims"]:
        np.random.seed(iii)
    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, array_list, lmax, dtype="complex64")

    log.info(f"[Sim n° {iii}] Generate signal alms in {time.time()-t0:.2f} s")
    master_alms = {}

    for sv in surveys:

        t1 = time.time()

        signal_alms = {}
        for ar in arrays[sv]:

            signal_alms[ar] = alms_cmb + fglms[f"{sv}_{ar}"]

            # Convolve the signal map with the pixwin only for CAR pixellization
            if d[f"pixwin_{sv}"]["pix"] == "CAR":
                # since the mnms noise sim include a pixwin, we convolve the signal ones
                signal = sph_tools.alm2map(signal_alms[ar], templates[sv])
                win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])
                signal = signal.convolve_with_pixwin(niter=niter, pixwin=pixwin[sv], window=win_kspace, use_ducc_rfft=True)
                signal_alms[ar] = sph_tools.map2alm(signal, niter, lmax)
            # Convolve the signal with the pixwin in harm. space for Planck sims
            if d[f"pixwin_{sv}"]["pix"] == "HEALPIX":
                signal_alms[ar] = curvedsky.almxfl(signal_alms[ar], pixwin[sv])

        log.info(f"[Sim n° {iii}] Convolve beam and pixwin in {time.time()-t1:.2f} s")

        wafers = sorted({ar.split("_")[0] for ar in arrays[sv]})

        for k in range(n_splits[sv]):
            for ar in arrays[sv]:

                l, bl = misc.read_beams(d[f"beam_T_{sv}_{ar}_per_split"][k], d[f"beam_pol_{sv}_{ar}_per_split"][k])
                s_alms = signal_alms[ar].copy()
                s_alms = misc.apply_beams(s_alms, bl)

                t3 = time.time()

                win_T = so_map.read_map(d[f"window_T_{sv}_{ar}_per_split"][k])
                win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}_per_split"][k])
                window_tuple = (win_T, win_pol)
                del win_T, win_pol

                log.info(f"[Sim n° {iii}] Read window in {time.time()-t3:.2f} s")

                noise_alms = np.load(f"{nlms_dir}/nlms_{sv}_{ar}_set{k}_{iii:05d}.npy")
                
                t4 = time.time()

                split = sph_tools.alm2map(s_alms + noise_alms, templates[sv])

                log.info(f"[Sim n° {iii}] alm2map for split {k} and array {ar} done in {time.time()-t4:.2f} s")

                t5 = time.time()
                if (window_tuple[0].pixel == "CAR") & (apply_kspace_filter):

                        win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}_per_split"][k])

                        inv_pwin = inv_pixwin[sv] if d[f"pixwin_{sv}"]["pix"] == "CAR" else None

                        split = kspace.filter_map(split,
                                                  filters[sv],
                                                  win_kspace,
                                                  inv_pixwin = inv_pwin,
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


        for s1 in range(n_splits[sv1]):
            for s2 in range(n_splits[sv2]):
                if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue

                mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}_{s1}x{sv2}_{ar2}_{s2}",
                                                    spin_pairs=spin_pairs)

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



                if apply_kspace_filter:
                    if kspace_tf_path == "analytical":
                        xtra_corr = None
                    else:
                        xtra_corr = TE_corr[f"{sv1}_{ar1}x{sv2}_{ar2}"]

                    lb, ps = kspace.deconvolve_kspace_filter_matrix(lb,
                                                                    ps,
                                                                    kspace_transfer_matrix[f"{sv1}_{ar1}x{sv2}_{ar2}"],
                                                                    spectra,
                                                                    xtra_corr=xtra_corr)

                if d[f"pixwin_{sv1}"]["pix"] == "HEALPIX":
                    _, xtra_pw1 = pspy_utils.naive_binning(np.arange(len(pixwin[sv1])), pixwin[sv1], binning_file, lmax)
                else:
                    xtra_pw1 = None
                if d[f"pixwin_{sv2}"]["pix"] == "HEALPIX":
                    _, xtra_pw2 = pspy_utils.naive_binning(np.arange(len(pixwin[sv2])), pixwin[sv2], binning_file, lmax)
                else:
                    xtra_pw2 = None
                lb, ps = transfer_function.deconvolve_xtra_tf(lb,
                                                              ps,
                                                              spectra,
                                                              xtra_pw1=xtra_pw1,
                                                              xtra_pw2=xtra_pw2)

                if write_all_spectra:
                    so_spectra.write_ps(sim_spec_dir + f"/{spec_name}_%05d.dat" % (iii), lb, ps, type, spectra=spectra)

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
        so_spectra.write_ps(sim_spec_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)

        if sv1 == sv2:
            spec_name_auto = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_auto_{iii:05d}"
            so_spectra.write_ps(sim_spec_dir + f"/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
            spec_name_noise = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_noise_{iii:05d}"
            so_spectra.write_ps(sim_spec_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)

    log.info(f"[Sim n° {iii}] Spectra computation done in {time.time()-t7:.2f} s")
    log.info(f"[Sim n° {iii}] Done in {time.time()-t0:.2f} s")
