"""
This script combine the cross array spectra into cross freq spectra that could be used in the likelihood
it also combine together all TE and EE spectra (since fg are subdominant)
"""

import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils, so_spectra, so_cov
from itertools import combinations_with_replacement as cwr
from itertools import product
from pspipe_utils import covariance, pspipe_list
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
spec_dir = "spectra"
combine_dir = "combined_spectra"
plot_dir = "plots/combined_cov"

pspy_utils.create_directory(combine_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_hi)


if d["cov_T_E_only"] == True:
    modes_for_xar_cov = ["TT", "TE", "ET", "EE"]
    modes_for_xfreq_cov = ["TT", "TE", "EE"]
else:
    modes_for_xar_cov = spectra
    modes_for_xfreq_cov = ["TT", "TE", "TB", "EE", "EB", "BB"]
    

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, char="_", return_nu_tag=True)
freq_list = pspipe_list.get_freq_list(d)

x_ar_cov_list = pspipe_list.x_ar_cov_order(spec_name_list, nu_tag_list, spectra_order=modes_for_xar_cov)
x_freq_cov_list = pspipe_list.x_freq_cov_order(freq_list, spectra_order=modes_for_xfreq_cov)
final_cov_list = pspipe_list.final_cov_order(freq_list, spectra_order=modes_for_xfreq_cov)



inv_cov, cov, vec = {}, {}, {}
combin_level = ["xar", "xfreq", "final"]
cov["xar"] = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                          cov_dir,
                                                          "analytic_cov",
                                                          spectra_order=modes_for_xar_cov,
                                                          remove_doublon=True,
                                                          check_pos_def=True)

inv_cov["xar"] = np.linalg.inv(cov["xar"])

vec["xar"] = covariance.read_x_ar_spectra_vec(spec_dir,
                                              spec_name_list,
                                              "cross",
                                              spectra_order = modes_for_xar_cov,
                                              type="Dl")

P_mat = covariance.get_x_ar_to_x_freq_P_mat(x_ar_cov_list,
                                            x_freq_cov_list,
                                            binning_file,
                                            lmax)
                                            
cov["xfreq"] = covariance.get_max_likelihood_cov(P_mat,
                                                 inv_cov["xar"],
                                                 force_sim = True,
                                                 check_pos_def = True)

inv_cov["xfreq"] = np.linalg.inv(cov["xfreq"])

vec["xfreq"]= covariance.max_likelihood_spectra(cov["xfreq"],
                                                inv_cov["xar"],
                                                P_mat,
                                                vec["xar"])

P_final = covariance.get_x_freq_to_final_P_mat(x_freq_cov_list,
                                               final_cov_list,
                                               binning_file,
                                               lmax)
                                               
cov["final"]  = covariance.get_max_likelihood_cov(P_final,
                                                  inv_cov["xfreq"],
                                                  force_sim = True,
                                                  check_pos_def = True)
                                                  
inv_cov["final"] = np.linalg.inv(cov["final"])

vec["final"] = covariance.max_likelihood_spectra(cov["final"],
                                                inv_cov["xfreq"],
                                                P_final,
                                                vec["xfreq"])


covariance.plot_P_matrix(P_mat, x_freq_cov_list, x_ar_cov_list, file_name=f"{combine_dir}/P_mat_x_ar_to_x_freq")
covariance.plot_P_matrix(P_final, final_cov_list, x_freq_cov_list, file_name=f"{combine_dir}/P_mat_x_freq_to_final")

for comb in combin_level:
    corr = so_cov.cov2corr(cov[comb], remove_diag=True)
    so_cov.plot_cov_matrix(corr, file_name=f"{combine_dir}/corr_{comb}")



def select_spec(spec_vec, cov, id, n_bins):
    ps = spec_vec[id * n_bins: (id + 1) * n_bins]
    cov_block = cov[id * n_bins: (id + 1) * n_bins, id * n_bins: (id + 1) * n_bins]
    std = np.sqrt(cov_block.diagonal())
    return ps, std

lscaling = {}
lscaling["TT"] = 2
lscaling["TE"] = 0
lscaling["TB"] = 0
lscaling["EE"] = 0
lscaling["EB"] = -0.5
lscaling["BB"] = -0.5

for spec in modes_for_xfreq_cov:
        x_freq_list = []
        if spec[0] == spec[1]:
            x_freq_list += [(f0, f1) for f0, f1 in cwr(freq_list, 2)]
        else:
            x_freq_list +=  [(f0, f1) for f0, f1 in product(freq_list, freq_list)]
        for x_freq in x_freq_list:
        
            plt.figure(figsize=(12,8))
    
            for id_ar, x_ar_cov_el in enumerate(x_ar_cov_list):
                spec1, name, nu_pair1 = x_ar_cov_el
                if (spec1 == spec) and (spec1[0] == spec1[1]):
                    if (x_freq == nu_pair1) or (x_freq == nu_pair1[::-1]):
                        ps, std = select_spec(vec["xar"], cov["xar"], id_ar, n_bins)
                        plt.errorbar(lb, ps * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_ar_cov_el , fmt=".")
                if (spec1 == spec) and (spec1[0] != spec1[1]) :
                    if (x_freq == nu_pair1):
                        ps, std = select_spec(vec["xar"], cov["xar"], id_ar, n_bins)
                        plt.errorbar(lb, ps * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_ar_cov_el, fmt=".")
                if (spec1[0] != spec1[1]) and (spec1 == spec[::-1]):
                    if (x_freq == nu_pair1[::-1]):
                        ps, std = select_spec(vec["xar"], cov["xar"], id_ar, n_bins)
                        plt.errorbar(lb, ps * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_ar_cov_el, fmt=".")

            for id_freq, x_freq_cov_el in enumerate(x_freq_cov_list):
                spec2, nu_pair2 = x_freq_cov_el
                if (spec2 == spec) and (x_freq == nu_pair2):
                    
                    ps, std = select_spec(vec["xfreq"], cov["xfreq"], id_freq, n_bins)
                    np.savetxt(f"{combine_dir}/{spec2}_xfreq_{x_freq[0]}x{x_freq[1]}.dat", np.transpose([lb, ps, std]))
                    plt.errorbar(lb, ps * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_freq_cov_el)

            plt.legend()
            plt.xlabel(r"$\ell$", fontsize=12)
            plt.ylabel(r"$\ell^{%.1f} D^{%s}_\ell$" % (lscaling[spec], spec), fontsize=20)
            plt.savefig(f"{combine_dir}/{spec}_xar_and_xfreq_{x_freq[0]}x{x_freq[1]}.png", bbox_inches="tight")
            plt.clf()
            plt.close()

        if spec == "TT": continue
        plt.figure(figsize=(12,8))
        for id_freq, x_freq_cov_el in enumerate(x_freq_cov_list):
            spec2, nu_pair2 = x_freq_cov_el
            if (spec2 == spec):
                for x_freq in x_freq_list:
                    if (x_freq == nu_pair2):
                        if 220 in x_freq: continue #remove 220 because it's scatter dominate the plot
                        ps, std = select_spec(vec["xfreq"], cov["xfreq"], id_freq, n_bins)
                        
                        plt.errorbar(lb, ps * lb ** lscaling[spec], std * lb ** lscaling[spec], label=x_freq_cov_el)

        for id_final, final_cov_el in enumerate(final_cov_list):
            spec3, _ = final_cov_el
            if spec3 == spec:
                ps, std = select_spec(vec["final"], cov["final"], id_final, n_bins)
                np.savetxt(f"{combine_dir}/{spec3}_final.dat", np.transpose([lb, ps, std]))
            
                plt.errorbar(lb, ps * lb ** lscaling[spec], std * lb ** lscaling[spec], label=spec + " final")
        
        plt.legend()
        plt.xlabel(r"$\ell$", fontsize=12)
        plt.ylabel(r"$\ell^{%.1f} D^{%s}_\ell$" % (lscaling[spec], spec), fontsize=20)
        plt.savefig(f"{combine_dir}/{spec}_xfreq_and_final.png", bbox_inches="tight")
        plt.clf()
        plt.close()
    
