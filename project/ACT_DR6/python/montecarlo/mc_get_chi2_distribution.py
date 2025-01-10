"""
This script compute the montecarlo chi2 distribution using the estimated simulation power spectra, the covariance matrix and the input theory.
We use the nominal ell cut of ACT DR6.
if the name of the folder containing the simulation spectra contains the string "_syst", we will use a covariance
with S+N+beam+leakage beam, otherwise we use simply the S+N covariance.
"""

import numpy as np
import pylab as plt
from pspipe_utils import covariance, pspipe_list, log
from pspy import so_cov, so_dict, pspy_utils, so_spectra
from pixell import utils
import sys
import scipy.stats as stats

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)
    
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
iStart = d["iStart"]
iStop = d["iStop"]

cov_dir = "covariances"
bestfit_dir = "best_fits"
sim_spec_dir = d["sim_spec_dir"]
mcm_dir = "mcms"


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


plot_dir = f"plots/chi2_distrib{add_str}"


pspy_utils.create_directory(plot_dir)
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

x_ar_cov = np.load(f"{cov_dir}/x_ar_final_cov_sim_gp.npy")
if include_syst:
    x_ar_beam_cov = np.load("covariances/x_ar_beam_cov.npy")
    x_ar_leakage_cov = np.load("covariances/x_ar_leakage_cov.npy")
    x_ar_cov += x_ar_beam_cov + x_ar_leakage_cov


selected_spectra = [spectra, ["TT", "TE", "ET", "EE"], ["TT"], ["TE"], ["ET"], ["TB"], ["BT"], ["EE"], ["EB"], ["BE"], ["BB"]]
name_list = ["all", "TT-TE-ET-EE", "TT", "TE", "ET", "TB", "BT", "EE", "EB", "BE", "BB"]

# Note that to match the null test selection and likelihood selection, we use a slightly different lmin
# This is because the likelihood cut based on the bin center, while most of the power spectrum pipeline cut
# based on the bin edges.
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[975, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[775, lmax], P=[475, lmax]),
    "dr6_pa6_f150": dict(T=[575, lmax], P=[475, lmax]),
    "dr6_pa5_f090": dict(T=[975, lmax], P=[475, lmax]),
    "dr6_pa6_f090": dict(T=[975, lmax], P=[475, lmax]),
}
only_TT_map_set = ["dr6_pa4_f220"]

theory_vec = covariance.read_x_ar_theory_vec(bestfit_dir, mcm_dir, spec_name_list, lmax, spectra_order=spectra)

for name, select in zip(name_list, selected_spectra):

    bin_out_dict, indices = covariance.get_indices(bin_low,
                                                   bin_high,
                                                   bin_mean,
                                                   spec_name_list,
                                                   spectra_cuts=spectra_cuts,
                                                   spectra_order=spectra,
                                                   selected_spectra=select,
                                                   only_TT_map_set=only_TT_map_set)
                                                   
                                                   
                                                   

    # some plot to check that the selection worked, we use sim 00000
    spec_plot_dir = f"{plot_dir}/{name}"
    pspy_utils.create_directory(spec_plot_dir)
    
    data_vec = covariance.read_x_ar_spectra_vec(sim_spec_dir, spec_name_list, f"cross_00000", spectra_order=spectra, type=type)
    sub_data_vec = data_vec[indices]
    sub_theory_vec = theory_vec[indices]
    sub_cov = x_ar_cov[np.ix_(indices,indices)]

    for my_spec in bin_out_dict.keys():
        s_name, spectrum = my_spec
        id, lb = bin_out_dict[my_spec]

        lb_, Db = so_spectra.read_ps(f"{sim_spec_dir}/Dl_{s_name}_cross_00000.dat", spectra=spectra)
        
        plt.figure(figsize=(12,8))
        plt.title(f"{my_spec}, min={np.min(lb)}, max={np.max(lb)}")
        if spectrum == "TT":
            plt.semilogy()
        plt.plot(lb_, Db[spectrum], label="original spectrum")
        plt.errorbar(lb, sub_data_vec[id], np.sqrt(sub_cov[np.ix_(id,id)].diagonal()), fmt=".", label="selected spectrum")
        plt.plot(lb, sub_theory_vec[id], "--", color="gray",alpha=0.3, label="theory")
        plt.legend()
        plt.savefig(f"{spec_plot_dir}/{spectrum}_{s_name}.png", bbox_inches="tight")
        plt.clf()
        plt.close()


    inv_sub_cov = np.linalg.inv(x_ar_cov[np.ix_(indices,indices)])

    chi2_list = []
    for iii in range(iStart, iStop + 1):
        data_vec = covariance.read_x_ar_spectra_vec(sim_spec_dir, spec_name_list, f"cross_{iii:05d}", spectra_order=spectra, type=type)
        res = data_vec[indices] - theory_vec[indices]
        chi2 = res @ inv_sub_cov  @ res
        chi2_list += [chi2]
        log.info(f"{name} Sim number, {iii}, chi2 = {chi2:.3f}, dof ={len(res)}")

    plt.figure(figsize=(12,8))
    plt.hist(chi2_list, bins=40, density=True, histtype="step", label="sims chi2 distribution")
    x = np.arange(len(res) - 400, len(res) + 400)
    plt.title(name)
    plt.plot(
        x,
        stats.chi2.pdf(x, df=len(res)),
        "-",
        linewidth=2,
        color="orange",
        label="expected distribution",
    )
    plt.savefig(f"{plot_dir}/histo_{name}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
