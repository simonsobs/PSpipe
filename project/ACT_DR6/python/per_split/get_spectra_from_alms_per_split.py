"""
This script compute all power spectra and write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the pixel window function.
The spectra are then combined in mean auto, cross and noise power spectrum and written to disk.
If write_all_spectra=True, each individual spectrum is also written to disk.
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


def bin_spectra_alaPlanck(l, cl, binning_file, lmax, type, spectra=None, mbb_inv=None, binned_mcm=True):

    """Bin the power spectra following planck analysis
    denoting f = l * (l+1)/(2*pi)
    Planck create Cl_b = \sum_{l \in b} (C_l * f_L) / \sum_{l \in b}  f_l
    not entirely sure why
    The reason the routine lives here is because this script is used to reproduce planck spectra
    
    Parameters
    ----------
    l: 1d array
      the multipoles
    cl: 1d array or dict of 1d array
      the power spectra to bin, can be a 1d array (spin0) or a dictionnary (spin0 and spin2)
    binning_file: data file
      a binning file with format bin low, bin high, bin mean
    lmax: int
      the maximum multipole to consider
    type: string
      the type of binning, either bin Cl or bin Dl
    spectra: list of string
      needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    mbb_inv: 2d array
      optionnaly apply the inverse of the  mode coupling matrix to debiais the spectra
    binned_mcm: boolean
      if mbb_inv is not None, specify if it's binned or not
      
    Return
    ----------
    The function return the binned multipole array bin_c and a 1d power spectrum
    array (or dictionnary of 1d power spectra if spectra is not None).
    """

    bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)

    # the alm2cl return cl starting at l = 0, we use spectra from l = 2
    # this is due in particular to the fact that the mcm is computed only for l>=2
    
    l = np.arange(2, lmax)
    if spectra is None: cl = cl[l]
    else: cl = {f: cl[f][l] for f in spectra}
    
    if type == "Dl": fac = (l * (l + 1) / (2 * np.pi))
    elif type == "Cl": fac = l * 0 + 1

    
    assert binned_mcm == False, "Planck deconvolve unbinned mcm"
    assert mbb_inv is not None, "you need to pass a mode_coupling matrix"

    l, cl = so_spectra.deconvolve_mode_coupling_matrix(l, cl, mbb_inv, spectra)
    
    # Now the binning part
    if spectra is None:
        ps = np.zeros(n_bins)
        for ibin in range(n_bins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
            ps[ibin] = (cl[loc] * fac[loc]).sum() / fac[loc].sum()
    else:
        vec = []
        for f in spectra:
            binned_power = np.zeros(n_bins)
            for ibin in range(n_bins):
                loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
                binned_power[ibin] = (cl[f][loc] * fac[loc]).sum()  / fac[loc].sum()
            vec = np.append(vec, binned_power)
        ps = so_spectra.vec2spec_dict(n_bins, vec, spectra)

    # still to convert with Dl, but with a binned vector, no way to do otherwise
    if type == "Dl": fac = (lb * (lb + 1) / (2 * np.pi))
    elif type == "Cl": fac = lb * 0 + 1

    for f in spectra:
        ps[f] *= fac

    return lb, ps
    

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
deconvolve_pixwin = d["deconvolve_pixwin"]
binned_mcm = d["binned_mcm"]
apply_kspace_filter = d["apply_kspace_filter"]
cov_T_E_only = d["cov_T_E_only"]
planck_like_binning = d.get("use_planck_like_binning", False)

log.info(f"use planck-like binning : {planck_like_binning}")



spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


mcm_dir = "mcms"
spec_dir = "spectra"
alms_dir = "alms"

pspy_utils.create_directory(spec_dir)

master_alms, nsplit, arrays, templates, filter_dicts = {}, {}, {}, {}, {}
# read the alms

for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    if apply_kspace_filter == True: filter_dicts[sv] = d[f"k_filter_{sv}"]
    templates[sv] = so_map.read_map(d[f"window_T_{sv}_{arrays[sv][0]}_per_split"][0])

    for ar in arrays[sv]:
        maps = d[f"maps_{sv}_{ar}"]
        nsplit[sv] = len(maps)
        log.info(f"{nsplit[sv]} split of survey: {sv}, array {ar}")
        for k, map in enumerate(maps):
            master_alms[sv, ar, k] = np.load(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy")
            log.debug(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy")

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

    nsplits_1 = nsplit[sv1]
    nsplits_2 = nsplit[sv2]

    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]


    for s1 in range(nsplits_1):
        for s2 in range(nsplits_2):
            if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue

            mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}_{s1}x{sv2}_{ar2}_{s2}",
                                                spin_pairs=spin_pairs)

            l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, s1],
                                                         master_alms[sv2, ar2, s2],
                                                         spectra=spectra)

            spec_name=f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_{s1}{s2}"

            if planck_like_binning == True:
                lb, ps = bin_spectra_alaPlanck(l,
                                               ps_master,
                                               binning_file,
                                               lmax,
                                               type=type,
                                               spectra=spectra,
                                               mbb_inv=mbb_inv,
                                               binned_mcm=binned_mcm)
            
            else:
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

            if write_all_spectra:
                so_spectra.write_ps(spec_dir + f"/{spec_name}.dat", lb, ps, type, spectra=spectra)

            for count, spec in enumerate(spectra):
                if (s1 == s2) & (sv1 == sv2):
                    if count == 0: log.debug(f"[{task}] auto {sv1}_{ar1} x {sv2}_{ar2} {s1}{s2}")
                    ps_dict[spec, "auto"] += [ps[spec]]
                else:
                    if count == 0: log.debug(f"[{task}] cross {sv1}_{ar1} x {sv2}_{ar2} {s1}{s2}")
                    ps_dict[spec, "cross"] += [ps[spec]]

    ps_dict_auto_mean = {}
    ps_dict_cross_mean = {}
    ps_dict_noise_mean = {}

    for spec in spectra:
        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)
        spec_name_cross = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_cross"

        if ar1 == ar2 and sv1 == sv2:
            # Average TE / ET so that for same array same season TE = ET
            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.

        if sv1 == sv2:
            ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
            spec_name_auto = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_auto"
            ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplit[sv1]
            spec_name_noise = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_noise"

    so_spectra.write_ps(spec_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)
    if sv1 == sv2:
        so_spectra.write_ps(spec_dir + f"/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
        so_spectra.write_ps(spec_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)
