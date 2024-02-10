"""
This script combine the different covariances matrix together
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
nkeep = 70

cov_dir = "covariances"
bestfit_dir = "best_fits"
sim_spec_dir = "sim_spectra"
mcm_dir = "mcms"
plot_dir = "plots/full_covariance"

pspy_utils.create_directory(plot_dir)

bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

x_ar_analytic_cov = np.load(f"{cov_dir}/x_ar_analytic_cov.npy")
x_ar_mc_cov = np.load(f"{cov_dir}/x_ar_mc_cov.npy")
x_ar_beam_cov = np.load(f"{cov_dir}/x_ar_beam_cov.npy")
x_ar_leakage_cov = np.load(f"{cov_dir}/x_ar_leakage_cov.npy")

x_ar_cov =  covariance.correct_analytical_cov_skew(x_ar_analytic_cov, x_ar_mc_cov, nkeep=nkeep)

np.save(f"{cov_dir}/x_ar_final_cov_sim.npy", x_ar_cov)

full_x_ar_cov = x_ar_cov + x_ar_beam_cov + x_ar_leakage_cov

np.save(f"{cov_dir}/x_ar_final_cov_data.npy", full_x_ar_cov)


### plot and test covariance in the used ell range

selected_spectra = spectra
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[975, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[775, lmax], P=[475, lmax]),
    "dr6_pa6_f150": dict(T=[575, lmax], P=[475, lmax]),
    "dr6_pa5_f090": dict(T=[975, lmax], P=[475, lmax]),
    "dr6_pa6_f090": dict(T=[975, lmax], P=[475, lmax]),
}

bin_out_dict,  all_indices = covariance.get_indices(bin_low,
                                                    bin_high,
                                                    bin_mean,
                                                    spec_name_list,
                                                    spectra_cuts=spectra_cuts,
                                                    spectra_order=spectra,
                                                    selected_spectra=selected_spectra)


sub_full_x_ar_cov = full_x_ar_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_cov = x_ar_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_analytic_cov = x_ar_analytic_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_beam_cov = x_ar_beam_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_leakage_cov = x_ar_leakage_cov[np.ix_(all_indices, all_indices)]


log.info(f"test S+N cov + beam cov + leakage cov")

pspy_utils.is_symmetric(sub_full_x_ar_cov, tol=1e-7)
pspy_utils.is_pos_def(sub_full_x_ar_cov)


plt.figure(figsize=(28,8))

plt.title(r"$\Sigma^{\rm MC}/\Sigma^{\rm analytic}$", fontsize=22)

xlabel = []
xlabel_loc = []
for my_spec in bin_out_dict.keys():
    id_spec, lb_spec = bin_out_dict[my_spec]
    mean_id = np.mean(id_spec)
    xlabel_loc += [mean_id]
    xlabel += [my_spec]
    plt.axvline(np.max(id_spec) + 0.5, color="black", alpha=0.4, linestyle="--")
    
plt.plot(sub_x_ar_cov.diagonal() / sub_x_ar_analytic_cov.diagonal())
plt.xticks(xlabel_loc, xlabel, rotation=90)
plt.tight_layout()
plt.savefig(f"{plot_dir}/MC_vs_analytic_cov.png")
plt.clf()
plt.close()


plt.figure(figsize=(28,8))

for my_spec in bin_out_dict.keys():
    id_spec, lb_spec = bin_out_dict[my_spec]
    plt.axvline(np.max(id_spec) + 0.5, color="black", alpha=0.4, linestyle="--")

plt.title(r"$\Sigma^{\rm comp}/\Sigma^{\rm tot}$", fontsize=22)
plt.plot(sub_x_ar_leakage_cov.diagonal()/sub_full_x_ar_cov.diagonal(), label="leakage cov / total cov")
plt.plot(sub_x_ar_beam_cov.diagonal()/sub_full_x_ar_cov.diagonal(), label="beam cov / total cov")
plt.xticks(xlabel_loc, xlabel, rotation=90)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f"{plot_dir}/cov_different_component.png")
plt.clf()
plt.close()
