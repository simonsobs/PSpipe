"""
This script combine the x-ar simulation spectra into x-freq and total (including all) spectra.
Note that we subtract the best fit fg model in order for all spectra to have the same expected mean.
if the name of the folder containing the simulation spectra contains the string "_syst", we will use a covariance
with S+N+beam+leakage beam for combining, otherwise we use simply the S+N covariance.
"""

from pspy import so_dict, pspy_utils
from pspipe_utils import covariance, pspipe_list, log
import numpy as np
import pylab as plt
import sys, os

def get_ml_bins(bin_out_dict, bin_mean):

    """
    Find the bin range of the maximum likelihood combination of the data
    we go through all spectra in the list and choose as a minimum bin the minimum bin
    of all spectra that enters the combinaison and as a maximum the maximum bin
    of all spectra that enters the combinaison
    """
    
    min_lb_list, max_lb_list = [], []
    for my_spec in bin_out_dict.keys():
        id_spec, lb_spec = bin_out_dict[my_spec]
        min_lb_list += [np.min(lb_spec)]
        max_lb_list += [np.max(lb_spec)]

    min_lb_comb = np.min(min_lb_list, axis=0)
    max_lb_comb = np.max(max_lb_list, axis=0)

    ml_id = np.where((bin_mean >= min_lb_comb) & (bin_mean <= max_lb_comb))
    ml_lb = bin_mean[ml_id]
    
    return ml_lb

def get_P_mat(vec_size, lb_ml, bin_out_dict):

    """
    Very naive "pointing" matrix for maximum likelihood combination of the spectra
    it should be vectorized but writing the loops was trivial and the code is sufficienctly fast
    """
    
    P_mat = np.zeros((vec_size, len(lb_ml)))
    
    index1 = 0
    y_ticks, y_name = [], []
    
    for my_spec in bin_out_dict.keys():
        s_name, _ = my_spec
        id_spec, lb_spec = bin_out_dict[my_spec]
        
        # for plotting
        y_ticks += [index1]
        y_name += [s_name]
        
        for jj in lb_spec:
            for index2, ii in enumerate(lb_ml):
                if ii == jj:
                    P_mat[index1, index2] = 1
            index1 += 1

    return P_mat
    
def combine_and_save_spectra(lb_ml, P_mat, cov_ml, i_sub_cov, sub_vec,  sub_vec_fg_th, name):
        
    vec_ml = covariance.max_likelihood_spectra(cov_ml,
                                               i_sub_cov,
                                               P_mat,
                                               sub_vec)

    sub_vec_fg_sub = sub_vec - sub_vec_fg_th
    vec_ml_fg_sub = covariance.max_likelihood_spectra(cov_ml,
                                                      i_sub_cov,
                                                      P_mat,
                                                      sub_vec_fg_sub)
    
    np.savetxt(f"{combined_spec_dir}/{type}_{name}_cmb_only.dat", np.transpose([lb_ml, vec_ml_fg_sub, np.sqrt(cov_ml.diagonal())]))

def ml_helper(x_ar_cov, vec_xar_th, vec_xar_fg_th, bin_mean, all_indices, bin_out_dict):

    sub_cov = x_ar_cov[np.ix_(all_indices, all_indices)]
    i_sub_cov = np.linalg.inv(sub_cov)
        
    sub_vec_th = vec_xar_th[all_indices]
    sub_vec_fg_th = vec_xar_fg_th[all_indices]
    
    lb_ml = get_ml_bins(bin_out_dict, bin_mean)
    P_mat = get_P_mat(len(sub_vec_fg_th), lb_ml, bin_out_dict)
        
    cov_ml = covariance.get_max_likelihood_cov(P_mat,
                                               i_sub_cov,
                                               force_sim = True,
                                               check_pos_def = False)

    vec_th_ml = covariance.max_likelihood_spectra(cov_ml,
                                                  i_sub_cov,
                                                  P_mat,
                                                  sub_vec_th)

    return lb_ml, P_mat, cov_ml, vec_th_ml, i_sub_cov,  sub_vec_fg_th


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

sim_spec_dir = d["sim_spec_dir"]

if "_syst" in sim_spec_dir:
    # so this is a bit dangerous but I need to know which covariance
    # to use when computing the chi2
    # usual sim don't include systematic and x_ar_final_cov_sim_gp.npy is appropriate
    # for sim with syst generated with mc_apply_syst_model we need to include beam and leakage beam cov
    # so I look for the string _syst in sim_spec_dir to know if systematic
    #have been included or not
    
    log.info(f"{sim_spec_dir} contains the string _syst, we will use covariance with beam and leakage beam")
    include_syst = True
    add_str = "_syst"
else:
    log.info(f"{sim_spec_dir} does not contains the string _syst, we will use covariance with only S+N")
    add_str = ""


combined_spec_dir = f"combined_sim_spectra{add_str}"
bestfit_dir = f"best_fits"
mcm_dir = "mcms"

pspy_utils.create_directory(combined_spec_dir)

binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
iStart = d["iStart"]
iStop =  d["iStop"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

x_ar_cov = np.load("covariances/x_ar_final_cov_sim_gp.npy")

if include_syst:
    x_ar_beam_cov = np.load("covariances/x_ar_beam_cov.npy")
    x_ar_leakage_cov = np.load("covariances/x_ar_leakage_cov.npy")
    x_ar_cov += x_ar_beam_cov + x_ar_leakage_cov

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

selected_spectra_list = [["TT"], ["TE", "ET"], ["TB", "BT"], ["EB", "BE"], ["EE"], ["BB"]]
only_TT_map_set = ["dr6_pa4_f220"]
########################################################################################


#### First start with the combination of all power spectra
#### we skip TT because the fg make different frequency spectra incompatible

print("")
print("all")

for spec_select in selected_spectra_list:
    spectrum = spec_select[0]
    name = f"all_{spectrum}"

    bin_out_dict,  all_indices = covariance.get_indices(bin_low,
                                                        bin_high,
                                                        bin_mean,
                                                        spec_name_list,
                                                        spectra_cuts=spectra_cuts,
                                                        spectra_order=spectra,
                                                        selected_spectra=spec_select,
                                                        excluded_map_set = None,
                                                        only_TT_map_set=only_TT_map_set)

    print("")
    print(f"{spec_select}, {list(bin_out_dict.keys())}")
    
    lb_ml, P_mat, cov_ml, vec_th_ml, i_sub_cov, sub_vec_fg_th = ml_helper(x_ar_cov, vec_xar_th, vec_xar_fg_th, bin_mean, all_indices, bin_out_dict)
    np.savetxt(f"{combined_spec_dir}/bestfit_{name}.dat", np.transpose([lb_ml, vec_th_ml]))
    np.save(f"{combined_spec_dir}/cov_{name}.npy", cov_ml)

    for iii in range(iStart, iStop + 1):
        sim_name = name + f"_{iii:05d}"
        print(sim_name)
        vec_xar = covariance.read_x_ar_spectra_vec(sim_spec_dir,
                                                   spec_name_list,
                                                   f"cross_{iii:05d}",
                                                   spectra_order=spectra,
                                                   type=type)

        sub_vec = vec_xar[all_indices]
        combine_and_save_spectra(lb_ml, P_mat, cov_ml, i_sub_cov, sub_vec,  sub_vec_fg_th, sim_name)



#### Now do the auto-freq spectra combination

auto_freq_pairs = ["150x150", "90x90", "220x220"]
excluded_map_set = {}
excluded_map_set["150x150"] = ["dr6_pa5_f090", "dr6_pa6_f090", "dr6_pa4_f220"]
excluded_map_set["90x90"] = ["dr6_pa5_f150", "dr6_pa6_f150", "dr6_pa4_f220"]
excluded_map_set["220x220"] = ["dr6_pa5_f090", "dr6_pa6_f090", "dr6_pa5_f150", "dr6_pa6_f150"]


print("")
print("auto-freq")

for spec_select in selected_spectra_list:
    spectrum = spec_select[0]

    for fp in auto_freq_pairs:
        if (fp == "220x220") & (spectrum != "TT"): continue
        
        name = f"{fp}_{spectrum}"

        bin_out_dict,  all_indices = covariance.get_indices(bin_low,
                                                            bin_high,
                                                            bin_mean,
                                                            spec_name_list,
                                                            spectra_cuts=spectra_cuts,
                                                            spectra_order=spectra,
                                                            selected_spectra=spec_select,
                                                            excluded_map_set = excluded_map_set[fp],
                                                            only_TT_map_set=only_TT_map_set)

        print("")
        print(f"{spec_select}, {fp}, {list(bin_out_dict.keys())}")
        lb_ml, P_mat, cov_ml, vec_th_ml, i_sub_cov,  sub_vec_fg_th = ml_helper(x_ar_cov, vec_xar_th, vec_xar_fg_th, bin_mean, all_indices, bin_out_dict)
        np.savetxt(f"{combined_spec_dir}/bestfit_{name}.dat", np.transpose([lb_ml, vec_th_ml]))
        np.save(f"{combined_spec_dir}/cov_{name}.npy", cov_ml)

        for iii in range(iStart, iStop + 1):
            sim_name = name + f"_{iii:05d}"
            print(sim_name)
            vec_xar = covariance.read_x_ar_spectra_vec(sim_spec_dir,
                                                       spec_name_list,
                                                       f"cross_{iii:05d}",
                                                       spectra_order=spectra,
                                                       type=type)

            sub_vec = vec_xar[all_indices]
            combine_and_save_spectra(lb_ml, P_mat, cov_ml, i_sub_cov, sub_vec,  sub_vec_fg_th, sim_name)

#### Now do the cross-freq spectra combination

cross_freq_pairs = ["90x220", "90x150", "150x220"]

map_set_A, map_set_B = {}, {}

map_set_A["90x150"] = ["dr6_pa5_f090", "dr6_pa6_f090"]
map_set_A["90x220"] = ["dr6_pa5_f090", "dr6_pa6_f090"]
map_set_A["150x220"] = ["dr6_pa5_f150", "dr6_pa6_f150"]
map_set_B["90x150"] = ["dr6_pa5_f150", "dr6_pa6_f150"]
map_set_B["90x220"] = ["dr6_pa4_f220"]
map_set_B["150x220"] = ["dr6_pa4_f220"]

print("")
print("x-freq")
for spec_select in selected_spectra_list:
    spectrum = spec_select[0]
    for fp in cross_freq_pairs:
        if ("220" in fp) & (spectrum != "TT"): continue
        name = f"{fp}_{spectrum}"

        bin_out_dict,  all_indices = covariance.get_cross_indices(bin_low,
                                                                  bin_high,
                                                                  bin_mean,
                                                                  spec_name_list,
                                                                  map_set_A = map_set_A[fp],
                                                                  map_set_B = map_set_B[fp],
                                                                  spectra_cuts=spectra_cuts,
                                                                  spectra_order=spectra,
                                                                  selected_spectra=spec_select,
                                                                  only_TT_map_set=only_TT_map_set)

        lb_ml, P_mat, cov_ml, vec_th_ml, i_sub_cov,  sub_vec_fg_th = ml_helper(x_ar_cov, vec_xar_th, vec_xar_fg_th, bin_mean, all_indices, bin_out_dict)
        np.savetxt(f"{combined_spec_dir}/bestfit_{name}.dat", np.transpose([lb_ml, vec_th_ml]))
        np.save(f"{combined_spec_dir}/cov_{name}.npy", cov_ml)
        for iii in range(iStart, iStop + 1):
            sim_name = name + f"_{iii:05d}"
            print(sim_name)
            vec_xar = covariance.read_x_ar_spectra_vec(sim_spec_dir,
                                                       spec_name_list,
                                                       f"cross_{iii:05d}",
                                                       spectra_order=spectra,
                                                       type=type)

            sub_vec = vec_xar[all_indices]
            combine_and_save_spectra(lb_ml, P_mat, cov_ml, i_sub_cov, sub_vec,  sub_vec_fg_th, sim_name)

        