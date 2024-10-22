"""
This script combine the montecarlo simulation spectra into single CMB only spectra
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

def get_P_mat(vec_size, lb_ml, bin_out_dict, spec_select, fig_name):

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

        plt.figure(figsize=(12,8))
        plt.imshow(P_mat, aspect="auto")
        plt.title(spec_select)
        plt.yticks(y_ticks, y_name)
        plt.xticks(np.arange(len(lb_ml))[::2], lb_ml[::2], rotation=90)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.clf()
        plt.close()
        
    return P_mat
    
        



d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)



spec_dir = f"sim_spectra"
bestfit_dir = f"best_fits"
mcm_dir = "mcms"
combined_spec_dir = "sim_combined_spectra"
plot_dir = f"plots/sim_combined_spectra/"

pspy_utils.create_directory(combined_spec_dir)
pspy_utils.create_directory(plot_dir)

binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

cov = np.load("covariances/x_ar_final_cov_sim_gp.npy")



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

iStart = d["iStart"]
iStop = d["iStop"]


for spec_select in selected_spectra_list:
    name = "all"
    spectrum = spec_select[0]
    if (spectrum == "TT"): continue
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
    
    sub_cov = cov[np.ix_(all_indices, all_indices)]
    i_sub_cov = np.linalg.inv(sub_cov)
    sub_vec_th = vec_xar_th[all_indices]
    lb_ml = get_ml_bins(bin_out_dict, bin_mean)
    P_mat = get_P_mat(len(sub_vec_th), lb_ml, bin_out_dict, spec_select, fig_name=f"{plot_dir}/P_mat_{name}_{spectrum}.png")

    cov_ml = covariance.get_max_likelihood_cov(P_mat,
                                               i_sub_cov,
                                               force_sim = True,
                                               check_pos_def = True)

    vec_th_ml = covariance.max_likelihood_spectra(cov_ml,
                                                  i_sub_cov,
                                                  P_mat,
                                                  sub_vec_th)
                       
    sigma_ml = np.sqrt(cov_ml.diagonal())
    
    np.savetxt(f"{combined_spec_dir}/bestfit_{name}_{spectrum}.dat", np.transpose([lb_ml, vec_th_ml]))
    np.save(f"{combined_spec_dir}/cov_{name}_{spectrum}.npy", cov_ml)

    for iii in range(iStart, iStop + 1):
        print(iii)

        vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                                   spec_name_list,
                                                   f"cross_{iii:05d}",
                                                   spectra_order=spectra,
                                                   type=type)

        sub_vec = vec_xar[all_indices]

        vec_ml = covariance.max_likelihood_spectra(cov_ml,
                                                   i_sub_cov,
                                                   P_mat,
                                                   sub_vec)

        np.savetxt(f"{combined_spec_dir}/{type}_{name}_{spectrum}_{iii:05d}.dat", np.transpose([lb_ml, vec_ml, sigma_ml]))

        vec_xar -= vec_xar_fg_th
        sub_vec = vec_xar[all_indices]
        vec_ml = covariance.max_likelihood_spectra(cov_ml,
                                                   i_sub_cov,
                                                   P_mat,
                                                   sub_vec)

        np.savetxt(f"{combined_spec_dir}/{type}_{name}_{spectrum}_{iii:05d}_cmb_only.dat", np.transpose([lb_ml, vec_ml, sigma_ml]))
