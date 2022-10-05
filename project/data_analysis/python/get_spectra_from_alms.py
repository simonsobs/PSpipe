"""
This script compute all power spectra and write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the pixel window function.
The spectra are then combined in mean auto, cross and noise power spectrum and written to disk.
If write_all_spectra=True, each individual spectrum is also written to disk.
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_map_preprocessing, so_mpi
from pspipe_utils import kspace, pspipe_list, transfer_function
from pixell import enmap
import numpy as np
import healpy as hp
import sys
import time


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
deconvolve_pixwin = d["deconvolve_pixwin"]
binned_mcm = d["binned_mcm"]
apply_kspace_filter = d["apply_kspace_filter"]

mcm_dir = "mcms"
spec_dir = "spectra"
alms_dir = "alms"

pspy_utils.create_directory(spec_dir)

master_alms, nsplit, arrays, templates, filter_dicts = {}, {}, {}, {}, {}
# read the alms

for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    if apply_kspace_filter == True: filter_dicts[sv] = d[f"k_filter_{sv}"]
    templates[sv] = so_map.read_map(d[f"window_T_{sv}_{arrays[sv][0]}"])
    
    for ar in arrays[sv]:
        maps = d[f"maps_{sv}_{ar}"]
        nsplit[sv] = len(maps)
        print(f"{nsplit[sv]} split of survey: {sv}, array {ar}")
        for k, map in enumerate(maps):
            master_alms[sv, ar, k] = np.load(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy")
            print(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy")
            
# compute the transfer functions
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(lb)
spec_name_list = pspipe_list.get_spec_name_list(d, char="_")

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
            kspace_transfer_matrix[spec_name] = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy", allow_pickle=True)
            
    
    # this will be used in the covariance computation
    for spec_name in spec_name_list:
        one_d_tf = kspace_transfer_matrix[spec_name].diagonal()[:4 * n_bins]
        np.savetxt(f"{spec_dir}/one_dimension_kspace_tf_{spec_name}.dat", one_d_tf)

# compute the power spectra

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
print(f"number of spectra to compute : {n_spec}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_spec - 1)

for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]
    
    xtra_pw1, xtra_pw2, mm_tf1, mm_tf2 = None, None, None, None
    if deconvolve_pixwin:
        # some xtra pixel window in the case of healpix
        # or Planck projected into CAR
        xtra_pw1 = transfer_function.healpix_pixwin(sv1, templates[sv1], binning_file, lmax)
        xtra_pw2 = transfer_function.healpix_pixwin(sv2, templates[sv2], binning_file, lmax)
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
    mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}",
                                        spin_pairs=spin_pairs)

                
    for s1 in range(nsplits_1):
        for s2 in range(nsplits_2):
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
            if apply_kspace_filter:
                lb, ps = kspace.deconvolve_kspace_filter_matrix(lb,
                                                                ps,
                                                                kspace_transfer_matrix[f"{sv1}_{ar1}x{sv2}_{ar2}"],
                                                                spectra)
                                                                
                                                            
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
                    if count == 0: print(f"auto {sv1}_{ar1} X {sv2}_{ar2} {s1}{s2}")
                    ps_dict[spec, "auto"] += [ps[spec]]
                else:
                    if count == 0:  print(f"cross {sv1}_{ar1} X {sv2}_{ar2} {s1}{s2}")
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


