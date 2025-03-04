"""
This script prepare the spectra for the release
"""

from pspy import so_dict, pspy_utils, so_spectra
from pspipe_utils import covariance, pspipe_list, log
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]
spec_dir = f"spectra_leak_corr_ab_corr_cal{tag}"
bestfit_dir = f"best_fits{tag}"
mcm_dir = "mcms"
release_spec_dir = f"release_spectra{tag}"
combined_spec_dir = f"combined_spectra{tag}"

pspy_utils.create_directory(f"{release_spec_dir}/array_bands")
pspy_utils.create_directory(f"{release_spec_dir}/freqs")
pspy_utils.create_directory(f"{release_spec_dir}/combined")

binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

# Check the binning
lb_, _  = so_spectra.read_ps(f"{spec_dir}/Dl_{spec_name_list[0]}_cross.dat")
assert (lb_ == bin_mean).all(), "binning file should be consistent with the one used to compute the spectra"

cov_xar = np.load("covariances/x_ar_final_cov_data.npy")

vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                           spec_name_list,
                                           "cross",
                                           spectra_order=spectra,
                                           type=type)

vec_xar_th = covariance.read_x_ar_theory_vec(bestfit_dir,
                                             mcm_dir,
                                             spec_name_list,
                                             lmax,
                                             spectra_order=spectra)

vec_xar_fg_th = covariance.read_x_ar_theory_vec(bestfit_dir,
                                                mcm_dir,
                                                spec_name_list,
                                                lmax,
                                                spectra_order=spectra,
                                                foreground_only=True)

########################################################################################
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[975, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[775, lmax], P=[775, lmax]),
    "dr6_pa6_f150": dict(T=[575, lmax], P=[575, lmax]),
    "dr6_pa5_f090": dict(T=[975, lmax], P=[975, lmax]),
    "dr6_pa6_f090": dict(T=[975, lmax], P=[975, lmax]),
}

release = "all_of_them"
if release == "likelihood":
    selected_spectra_list = ["TT", "TE", "ET", "EE"]
    my_spectra = ["TT", "TE", "EE"]

if release == "all_of_them":
    selected_spectra_list = None
    my_spectra = ["TT", "TE", "TB" ,"EE", "EB", "BB"]

only_TT_map_set = ["dr6_pa4_f220"]
########################################################################################

bin_out_dict, indices = covariance.get_indices(bin_low,
                                               bin_high,
                                               bin_mean,
                                               spec_name_list,
                                               spectra_cuts=spectra_cuts,
                                               spectra_order=spectra,
                                               selected_spectra= selected_spectra_list,
                                               excluded_map_set = None,
                                               only_TT_map_set=only_TT_map_set)

cov = cov_xar[np.ix_(indices, indices)]
vec = vec_xar[indices]
vec_th = vec_xar_th[indices]
vec_fg_th = vec_xar_fg_th[indices]


for my_spec in bin_out_dict.keys():
    s_name, spectrum = my_spec
    id, lb = bin_out_dict[my_spec]
    
    sub_cov = cov[np.ix_(id,id)]
    std = np.sqrt(sub_cov.diagonal())
    sub_vec, sub_vec_th, sub_vec_fg_th  = vec[id], vec_th[id], vec_fg_th[id]
    
    na, nb = s_name.split("x")
    
    if na == nb:
        my_str = "auto"
    else:
        my_str = "cross"

    comments = f"### \t ACT DR6 {my_str} array-band x-spectrum ### \n"
    header = "# bin_center,   Dl(data),     sigma(Dl(data)), Dl(best fit), Dl(best fit foreground)"
    
    np.savetxt(f"{release_spec_dir}/array_bands/{na}x{nb}_{spectrum}.dat", np.transpose([lb, sub_vec, std, sub_vec_th, sub_vec_fg_th]), fmt="%.8e", header=header, comments=comments)
    np.save(f"{release_spec_dir}/array_bands/cov_{na}x{nb}_{spectrum}.npy", sub_cov)

x_freq = ["90x90", "90x150", "150x150", "90x220", "150x220", "220x220"]
for xf in x_freq:
    for spectrum in my_spectra:
        if ("220" in xf) & (spectrum != "TT"): continue
        lb, Db, sigma_b = np.loadtxt(f"{combined_spec_dir}/Dl_{xf}_{spectrum}.dat", unpack=True)
        _, Db_cmb_only, _ = np.loadtxt(f"{combined_spec_dir}/Dl_{xf}_{spectrum}_cmb_only.dat", unpack=True)
        cov =  np.load(f"{combined_spec_dir}/cov_{xf}_{spectrum}.npy")
        
        comments = f"### \t ACT DR6 x-frequency spectrum ### \n"
        header = "# bin_center,   Dl(data),     sigma(Dl(data)), Dl(data, fg sub)"

        np.savetxt(f"{release_spec_dir}/freqs/{xf}_{spectrum}.dat", np.transpose([lb, Db, sigma_b, Db_cmb_only]), fmt="%.8e", header=header, comments=comments)
        np.save(f"{release_spec_dir}/freqs/cov_{xf}_{spectrum}.npy", cov)

for spectrum in my_spectra:
    lb, Db, sigma_b = np.loadtxt(f"{combined_spec_dir}/Dl_all_{spectrum}_cmb_only.dat", unpack=True)
    cov =  np.load(f"{combined_spec_dir}/cov_all_{spectrum}.npy")

    comments = f"### \t ACT DR6 combined spectrum ### \n"
    header = "# bin_center, Dl(data, fg sub), sigma(Dl(data))"
    np.savetxt(f"{release_spec_dir}/combined/fg_subtracted_{spectrum}.dat", np.transpose([lb, Db, sigma_b]), fmt="%.8e", header=header, comments=comments)
    np.save(f"{release_spec_dir}/combined/cov_{spectrum}.npy", cov)

os.system(f"cp {combined_spec_dir}/dataset_trace.pkl {release_spec_dir}/dataset_trace.pkl")
