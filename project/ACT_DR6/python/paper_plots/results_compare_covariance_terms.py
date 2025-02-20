"""
This script compare the magnitude of the different covariance contributions
"""

from pspy import so_dict, pspy_utils
from pspipe_utils import covariance, pspipe_list, log
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from matplotlib import rcParams

labelsize = 14
fontsize = 20

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

def get_P_mat(vec_size, lb_ml, bin_out_dict, fig_name=None):

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
            
    if fig_name is not None:

        plt.figure(figsize=(12,8))
        plt.imshow(P_mat, aspect="auto")
        plt.yticks(y_ticks, y_name)
        plt.xticks(np.arange(len(lb_ml))[::2], lb_ml[::2], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{fig_name}")
        plt.clf()
        plt.close()
        
    return P_mat
    

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

cov_sim = np.load("covariances/x_ar_final_cov_sim_gp.npy")
cov_term = {}
cov_term["tSZ"] = np.load("covariances/x_ar_non_gaussian_cov_tSZ.npy")
cov_term["radio"] = np.load("covariances/x_ar_non_gaussian_cov_radio.npy")
cov_term["lensing"] = np.load("covariances/x_ar_non_gaussian_cov_lensing.npy")
cov_term[r"T$\rightarrow$P leakage"] = np.load("covariances/x_ar_leakage_cov.npy")
cov_term["beam"] = np.load("covariances/x_ar_beam_cov.npy")
cov_term["CIB"] = np.load("covariances/x_ar_non_gaussian_cov_CIB.npy")

########################################################################################
# Note that for this plot we push the lmin a bit lower than for our nominal data cut
# This is to illustrate the magnitude of the errors at low ell

lmin = 575
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[lmin, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa6_f150": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa5_f090": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa6_f090": dict(T=[lmin, lmax], P=[lmin, lmax]),
}

selected_spectra_list = [["TT"], ["TE", "ET"], ["EE"]]
only_TT_map_set = ["dr6_pa4_f220"]
########################################################################################

#### Now do the auto-freq spectra combination

auto_freq_pairs = ["90x90", "150x150"]
excluded_map_set = {}
excluded_map_set["150x150"] = ["dr6_pa5_f090", "dr6_pa6_f090", "dr6_pa4_f220"]
excluded_map_set["90x90"] = ["dr6_pa5_f150", "dr6_pa6_f150", "dr6_pa4_f220"]
excluded_map_set["220x220"] = ["dr6_pa5_f090", "dr6_pa6_f090", "dr6_pa5_f150", "dr6_pa6_f150"]


ylim, yticks_loc, yticks_name = {}, {}, {}
for mode in ["TT", "TE", "EE"]:
    ylim[mode] = [10**-1, 100]
    yticks_loc[mode] = [1, 10, 100]
    yticks_name[mode] = ["1%", "10%", "100%"]


plt.figure(figsize=(10, 8), dpi=100)
plt.suptitle("Non-Gaussian and Systematic Error Budget", fontsize=fontsize, y=0.93)

count = 1
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
        
        lb_ml = get_ml_bins(bin_out_dict, bin_mean)

        cov_SN = cov_sim.copy()
        sub_cov = cov_SN[np.ix_(all_indices, all_indices)]
        
        i_sub_cov = np.linalg.inv(sub_cov)
    
        P_mat = get_P_mat(sub_cov.shape[0], lb_ml, bin_out_dict)
        cov_ml = covariance.get_max_likelihood_cov(P_mat,
                                                   i_sub_cov,
                                                   force_sim = True,
                                                   check_pos_def = False)

        sigma_ml_SN = np.sqrt(cov_ml.diagonal())

        plt.subplot(3,2,count)
        plt.semilogy()
        plt.hlines(1, 0, 10000, color="gray", alpha=0.4, linestyle=":")
        plt.hlines(10, 0, 10000, color="gray", alpha=0.4, linestyle=":")

        if count % 2 == 0:
            plt.yticks([])
        else:
            plt.yticks(yticks_loc[spectrum], yticks_name[spectrum])

        plt.ylim(ylim[spectrum])
       
        fa, fb = fp.split("x")
        plt.title(f"{spectrum} {fa} GHz x {fb} GHz", x=0.6, y=0.8, fontsize=labelsize)
        for term in cov_term.keys():
            my_cov = cov_sim.copy()
            my_cov += cov_term[term]
            sub_cov = my_cov[np.ix_(all_indices, all_indices)]
            i_sub_cov = np.linalg.inv(sub_cov)

            cov_ml = covariance.get_max_likelihood_cov(P_mat,
                                                       i_sub_cov,
                                                       force_sim = True,
                                                       check_pos_def = False)

            sigma_ml = np.sqrt(cov_ml.diagonal())
            plt.plot(lb_ml, (sigma_ml/sigma_ml_SN - 1) * 100, label=term)
        
        if count > 4:
            plt.xlabel(r"$\ell$", fontsize=fontsize)
            my_ticks = [1000, 2000,3000, 4000, 5000]
            plt.xticks(my_ticks, my_ticks)
        else:
            plt.xticks([])
        if count == 2:
            legend = plt.legend(bbox_to_anchor=(1.02, 0), fontsize=labelsize, title=r"$\bf{X}:$", title_fontsize=labelsize)
            legend._legend_box.align = "left"
        if count == 3:
            plt.ylabel(r"$\left(\sigma^{(\rm CV + noise + \bf{X})}/\sigma^{(\rm CV + noise)} - 1 \right) $", fontsize=fontsize)

        plt.xlim(0, 6000)
        plt.tick_params(labelsize=labelsize)
        count += 1

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f"{paper_plot_dir}/relative_contribution_to_errors.pdf", bbox_inches='tight')
plt.clf()
plt.close()
