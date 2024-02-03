"""
This script compute the montecarlo chi2 distribution using the estimted simulation power spectra, the covariance matrix and the input theory, we use the ell cut of ACT DR6
"""
import numpy as np
import pylab as plt
from pspipe_utils import covariance, pspipe_list, log
from pspy import so_cov, so_dict, pspy_utils
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
sim_spec_dir = "sim_spectra"
mcm_dir = "mcms"
plot_dir = "plots/chi2_distrib"

pspy_utils.create_directory(plot_dir)

bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

x_ar_cov = np.load(f"{cov_dir}/x_ar_final_cov_sim.npy")

selected_spectra = [spectra, ["TT", "TE", "ET", "EE"], ["TT"], ["TE"], ["ET"], ["TB"], ["BT"], ["EE"], ["EB"], ["BE"], ["BB"]]
name_list = ["all", "TT-TE-ET-EE", "TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[1000, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[800, lmax], P=[500, lmax]),
    "dr6_pa6_f150": dict(T=[600, lmax], P=[500, lmax]),
    "dr6_pa5_f090": dict(T=[1000, lmax], P=[500, lmax]),
    "dr6_pa6_f090": dict(T=[1000, lmax], P=[500, lmax]),
}

theory_vec = covariance.read_x_ar_theory_vec(bestfit_dir, mcm_dir, spec_name_list, lmax, spectra_order=spectra)

for name, select in zip(name_list, selected_spectra):

    bin_out_dict, indices = covariance.get_indices(bin_low,
                                                   bin_high,
                                                   bin_mean,
                                                   spec_name_list,
                                                   spectra_cuts=spectra_cuts,
                                                   spectra_order=spectra,
                                                   selected_spectra=select)

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
