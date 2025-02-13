"""
This script generate spectra from planck end-to-end simulation to get the correction to apply to Planck data
It's very similar to mnms_get_spectra_from_nlms but there is no signal added to the simulations
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi
import numpy as np
import sys
import time
import healpy as hp
from pixell import curvedsky
from pspipe_utils import simulation, pspipe_list, kspace, misc, log, transfer_function

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = ["Planck"]
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
spec_dir = "sim_spectra_planck_noise_and_syst"
nlms_dir = "noise_alms"

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




so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    t0 = time.time()
    
    master_alms = {}
    for sv in surveys:
        wafers = sorted({ar.split("_")[0] for ar in arrays[sv]})
        for k in range(n_splits[sv]):
            for ar in arrays[sv]:

                win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
                win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
                window_tuple = (win_T, win_pol)
                del win_T, win_pol

                noise_alms = np.load(f"{nlms_dir}/nlms_{sv}_{ar}_set{k}_{iii:05d}.npy")

                split = sph_tools.alm2map(noise_alms, templates[sv])

                if (window_tuple[0].pixel == "CAR") & (apply_kspace_filter):

                        win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])

                        inv_pwin = inv_pixwin[sv] if d[f"pixwin_{sv}"]["pix"] == "CAR" else None

                        split = kspace.filter_map(split,
                                                  filters[sv],
                                                  win_kspace,
                                                  inv_pixwin = inv_pwin,
                                                  weighted_filter=filter_dicts[sv]["weighted"],
                                                  use_ducc_rfft=True)


                        del win_kspace


                if d["remove_mean"] == True:
                    split = split.subtract_mean(window_tuple)

                master_alms[sv, ar, k] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)


    ps_dict = {}
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



                xtra_corr = None

                lb, ps = kspace.deconvolve_kspace_filter_matrix(lb,
                                                                ps,
                                                                kspace_transfer_matrix[f"{sv1}_{ar1}x{sv2}_{ar2}"],
                                                                spectra,
                                                                xtra_corr=None)

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
                    so_spectra.write_ps(spec_dir + f"/{spec_name}_%05d.dat" % (iii), lb, ps, type, spectra=spectra)

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


    log.info(f"[Sim n° {iii}] Done in {time.time()-t0:.2f} s")
