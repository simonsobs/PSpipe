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

use_mc_corrected_cov = True
only_diag_corrections = False

cov_name = "analytic_cov"
if use_mc_corrected_cov:
    if only_diag_corrections:
        cov_name += "_with_diag_mc_corrections"
    else:
        cov_name += "_with_mc_corrections"


cov_dir = "covariances"
spec_dir = "spectra"
like_product_dir = "like_product"
plot_dir = "plots/combined_cov"


pspy_utils.create_directory(like_product_dir)
pspy_utils.create_directory(plot_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, char="_", return_nu_tag=True)
spec_name_list_ET = pspipe_list.get_spec_name_list(d, char="_", remove_same_ar_and_sv=True)

freq_list = pspipe_list.get_freq_list(d)

fpairs = [f"{f0}x{f1}" for f0, f1 in cwr(freq_list, 2)]
fpairs_cross = [f"{f0}x{f1}" for f0, f1 in product(freq_list, freq_list)]

cov, vec = {}, {}

vec["xar"] = covariance.read_x_ar_spectra_vec(spec_dir,
                                             spec_name_list,
                                             "cross",
                                             spectra_order = ["TT", "TE", "ET", "EE"],
                                             type="Dl")

cov["xar"] = np.load(f"{like_product_dir}/x_ar_{cov_name}_with_beam.npy")
inv_cov_xar = np.linalg.inv(cov["xar"])

P_mat = covariance.get_x_ar_to_x_freq_P_mat(freq_list, spec_name_list, nu_tag_list, binning_file, lmax)
P_mat_cross = covariance.get_x_ar_to_x_freq_P_mat_cross(freq_list, spec_name_list, nu_tag_list, binning_file, lmax)
P_x_ar_to_x_freq = covariance.combine_P_mat(P_mat, P_mat_cross)
so_cov.plot_cov_matrix(P_x_ar_to_x_freq, file_name=f"{plot_dir}/P_mat_xar_to_xfreq")

cov["xfreq"] = covariance.get_max_likelihood_cov(P_x_ar_to_x_freq, inv_cov_xar, force_sim = True, check_pos_def = True)
vec["xfreq"] = covariance.max_likelihood_spectra(cov["xfreq"], inv_cov_xar, P_x_ar_to_x_freq, vec["xar"])

inv_cov_xfreq = np.linalg.inv(cov["xfreq"])
P_final = covariance.get_x_freq_to_final_P_mat(freq_list, binning_file, lmax)
so_cov.plot_cov_matrix(P_final, file_name=f"{plot_dir}/P_mat_final")

cov["final"]  = covariance.get_max_likelihood_cov(P_final, inv_cov_xfreq, force_sim = True, check_pos_def = True)
vec["final"] = covariance.max_likelihood_spectra(cov["final"], inv_cov_xfreq, P_final, vec["xfreq"])

spectra_order, block_order = {}, {}
spectra_order["xfreq"] = ["TT", "TE", "EE"]
spectra_order["final"] = ["TT", "TE", "EE"]

block_order["xfreq"] = {}
block_order["xfreq"]["TT"] = fpairs
block_order["xfreq"]["TE"] = fpairs_cross
block_order["xfreq"]["EE"] = fpairs

block_order["final"] = {}
block_order["final"]["TT"] = fpairs
block_order["final"]["TE"] = ["all"]
block_order["final"]["EE"] = ["all"]

combin_level = ["xfreq", "final"]
for comb in combin_level:
    corr = so_cov.cov2corr(cov[comb], remove_diag=True)
    so_cov.plot_cov_matrix(corr, file_name=f"{plot_dir}/corr_{comb}")
    np.save(f"{like_product_dir}/cov_{comb}.npy", cov[comb])
    lb, spec_dict, std_dict = covariance.from_vector_and_cov_to_ps_and_std_dict(vec[comb],
                                                                                cov[comb],
                                                                                ["TT", "TE", "EE"],
                                                                                block_order[comb],
                                                                                binning_file,
                                                                                lmax)

    for skip_220 in [True, False]:
        for spec in ["TT", "TE", "EE"]:
            plt.figure(figsize=(12, 8))
            for block in block_order[comb][spec]:
                D_b = spec_dict[spec, block]
                sigma_b = std_dict[spec, block]

                if skip_220 == False:
                    np.savetxt(f"{like_product_dir}/spectra_{comb}_{spec}_{block}.dat", np.transpose([lb, D_b, sigma_b]))
                else:
                    if "220" in block: continue
                plt.errorbar(lb, D_b, sigma_b, fmt=".", label=block)

            plt.legend()
            plt.savefig(f"{plot_dir}/{comb}_{spec}_{block}_skip220={skip_220}.png", bbox_inches="tight")
            plt.clf()
            plt.close()
