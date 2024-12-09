"""
This script compute all power spectra of all agora components and write them to disk
"""

import sys

import healpy as hp
import numpy as np
from pixell import enmap
from pspipe_utils import kspace, log, pspipe_list, transfer_function
from pspy import pspy_utils, so_dict, so_map, so_mcm, so_mpi, so_spectra

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
deconvolve_pixwin = d["deconvolve_pixwin"]
binned_mcm = d["binned_mcm"]
apply_kspace_filter = d["apply_kspace_filter"]
cov_T_E_only = d["cov_T_E_only"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
components = ["anomalous", "cib", "cmb_seed1", "dust", "ksz", "radio", "rksz", "sync", "tsz"]

mcm_dir = "mcms"
spec_dir = "spectra_components"
alms_dir = "alms_components"

pspy_utils.create_directory(spec_dir)

master_alms, arrays, templates, filter_dicts = {}, {}, {}, {}
# read the alms

for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    if apply_kspace_filter == True: filter_dicts[sv] = d[f"k_filter_{sv}"]
    templates[sv] = so_map.read_map(d[f"window_T_{sv}_{arrays[sv][0]}"])

    for ar in arrays[sv]:
        for k, comp in enumerate(components):
            master_alms[sv, ar, k] = np.load(f"{alms_dir}/alms_{sv}_{ar}_{comp}.npy")
            log.debug(f"{alms_dir}/alms_{sv}_{ar}_{comp}.npy")

# compute the transfer functions
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(lb)
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

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


    # this will be used in the covariance computation
    for spec_name in spec_name_list:
        one_d_tf = kspace_transfer_matrix[spec_name].diagonal()
        if cov_T_E_only == True: one_d_tf = one_d_tf[:4 * n_bins]
        np.savetxt(f"{spec_dir}/one_dimension_kspace_tf_{spec_name}.dat", one_d_tf)

# compute the power spectra


n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
log.info(f"number of spectra to compute : {n_spec}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_spec - 1)

for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]

    log.info(f"[{task}] Computing spectra for {sv1}_{ar1} x {sv2}_{ar2}")

    xtra_pw1, xtra_pw2, mm_tf1, mm_tf2 = None, None, None, None
    if deconvolve_pixwin:
        if d[f"pixwin_{sv1}"]["pix"] == "HEALPIX":
            pixwin_l = hp.pixwin(d[f"pixwin_{sv1}"]["nside"])
            lb, xtra_pw1 = pspy_utils.naive_binning(np.arange(len(pixwin_l)),  pixwin_l, binning_file, lmax)
        if d[f"pixwin_{sv2}"]["pix"] == "HEALPIX":
            pixwin_l = hp.pixwin(d[f"pixwin_{sv2}"]["nside"])
            lb, xtra_pw2 = pspy_utils.naive_binning(np.arange(len(pixwin_l)),  pixwin_l, binning_file, lmax)


    if d[f"deconvolve_map_maker_tf_{sv1}"]:
        mm_tf1 = so_spectra.read_ps(d[f"mm_tf_{sv1}_{ar1}.dat"], spectra=spectra)
    if d[f"deconvolve_map_maker_tf_{sv2}"]:
        mm_tf2 = so_spectra.read_ps(d[f"mm_tf_{sv2}_{ar2}.dat"], spectra=spectra)


    ps_dict = {}
    for spec in spectra:
        ps_dict[spec, "auto"] = []
        ps_dict[spec, "cross"] = []


    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
    mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}",
                                        spin_pairs=spin_pairs)


    for s1, comp1 in enumerate(components):
        for s2, comp2 in enumerate(components):

            l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, s1],
                                                         master_alms[sv2, ar2, s2],
                                                         spectra=spectra)

            spec_name=f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_{comp1}x{comp2}"

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


            lb, ps = transfer_function.deconvolve_xtra_tf(lb,
                                                          ps,
                                                          spectra,
                                                          xtra_pw1=xtra_pw1,
                                                          xtra_pw2=xtra_pw2,
                                                          mm_tf1=mm_tf1,
                                                          mm_tf2=mm_tf2)

            log.debug(f"[{task}] {sv1}_{ar1} x {sv2}_{ar2} {comp1}{comp2}")
            
            so_spectra.write_ps(spec_dir + f"/{spec_name}.dat", lb, ps, type, spectra=spectra)
