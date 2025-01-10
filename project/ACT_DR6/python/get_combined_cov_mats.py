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
import matplotlib

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "20"


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
use_gaussian_smoothing = d["use_gaussian_smoothing"]
nkeep = 70

cov_dir = "covariances"
plot_dir = "plots/full_covariance"

pspy_utils.create_directory(plot_dir)

bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

x_ar_analytic_cov = np.load(f"{cov_dir}/x_ar_analytic_cov.npy")
x_ar_mc_cov = np.load(f"{cov_dir}/x_ar_mc_cov.npy")
x_ar_beam_cov = np.load(f"{cov_dir}/x_ar_beam_cov.npy")
x_ar_leakage_cov = np.load(f"{cov_dir}/x_ar_leakage_cov.npy")
x_ar_non_gaussian_cov_radio = np.load(f"{cov_dir}/x_ar_non_gaussian_cov_radio.npy")
x_ar_non_gaussian_cov_tSZ = np.load(f"{cov_dir}/x_ar_non_gaussian_cov_tSZ.npy")
x_ar_non_gaussian_cov_lensing =  np.load(f"{cov_dir}/x_ar_non_gaussian_cov_lensing.npy")
x_ar_non_gaussian_cov_CIB =  np.load(f"{cov_dir}/x_ar_non_gaussian_cov_CIB.npy")

if use_gaussian_smoothing:
    log.info(f"we will use the analytical cov corrected from simulations using gaussian processes")
    x_ar_cov = np.load(f"{cov_dir}/x_ar_final_cov_sim_gp.npy")
else:
    log.info(f"we will use the skew method to corrected the analytical cov mat with simulation")
    x_ar_cov =  covariance.correct_analytical_cov_skew(x_ar_analytic_cov, x_ar_mc_cov, nkeep=nkeep)
    np.save(f"{cov_dir}/x_ar_final_cov_sim_skew.npy", x_ar_cov)

full_x_ar_cov = x_ar_cov + x_ar_beam_cov + x_ar_leakage_cov + x_ar_non_gaussian_cov_radio + x_ar_non_gaussian_cov_lensing + x_ar_non_gaussian_cov_tSZ + x_ar_non_gaussian_cov_CIB

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

only_TT_map_set = ["dr6_pa4_f220"]


bin_out_dict,  all_indices = covariance.get_indices(bin_low,
                                                    bin_high,
                                                    bin_mean,
                                                    spec_name_list,
                                                    spectra_cuts=spectra_cuts,
                                                    spectra_order=spectra,
                                                    selected_spectra=selected_spectra,
                                                    only_TT_map_set=only_TT_map_set)


sub_full_x_ar_cov = full_x_ar_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_cov = x_ar_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_analytic_cov = x_ar_analytic_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_beam_cov = x_ar_beam_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_leakage_cov = x_ar_leakage_cov[np.ix_(all_indices, all_indices)]
sub_x_ar_non_gaussian_cov_radio = x_ar_non_gaussian_cov_radio[np.ix_(all_indices, all_indices)]
sub_x_ar_non_gaussian_cov_lensing = x_ar_non_gaussian_cov_lensing[np.ix_(all_indices, all_indices)]
sub_x_ar_non_gaussian_cov_tSZ = x_ar_non_gaussian_cov_tSZ[np.ix_(all_indices, all_indices)]
sub_x_ar_non_gaussian_cov_CIB = x_ar_non_gaussian_cov_CIB[np.ix_(all_indices, all_indices)]

log.info(f"test S+N cov + beam cov + leakage cov + non gaussian cov")

pspy_utils.is_symmetric(sub_full_x_ar_cov, tol=1e-7)
pspy_utils.is_pos_def(sub_full_x_ar_cov)

plt.figure(figsize=(28,8))

plt.title(r"$\Sigma^{\rm MC}/\Sigma^{\rm analytic}$", fontsize=22)

name_list, label_loc = [], []
for my_spec in bin_out_dict.keys():
    id_spec, lb_spec = bin_out_dict[my_spec]
    mean_id = np.mean(id_spec)
    label_loc += [mean_id]
    
    name, spectrum = my_spec
    name = name.replace("dr6_", "")
    name_list += [f"{spectrum} {name}"]

    plt.axvline(np.max(id_spec) + 0.5, color="black", alpha=0.4, linestyle="--")
    
plt.plot(sub_x_ar_cov.diagonal() / sub_x_ar_analytic_cov.diagonal())
plt.xticks(label_loc, name_list, rotation=90)
plt.tight_layout()
plt.savefig(f"{plot_dir}/MC_vs_analytic_cov.png")
plt.clf()
plt.close()

if selected_spectra == ["TE", "ET"]:
    corr = so_cov.cov2corr(sub_x_ar_leakage_cov, remove_diag=False)
    plt.figure(figsize=(18,12))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(label_loc, name_list, rotation=90)
    plt.yticks(label_loc, name_list)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/corr_leakage.png")
    plt.clf()
    plt.close()

if selected_spectra == "TT":
    corr = so_cov.cov2corr(sub_x_ar_non_gaussian_cov_radio, remove_diag=False)
    plt.figure(figsize=(18,12))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(label_loc, name_list, rotation=90)
    plt.yticks(label_loc, name_list)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/radio_leakage.png")
    plt.clf()
    plt.close()

    corr = so_cov.cov2corr(sub_x_ar_non_gaussian_cov_tSZ, remove_diag=False)
    plt.figure(figsize=(18,12))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(label_loc, name_list, rotation=90)
    plt.yticks(label_loc, name_list)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/radio_tSZ.png")
    plt.clf()
    plt.close()


corr = so_cov.cov2corr(sub_full_x_ar_cov, remove_diag=False)
plt.figure(figsize=(18,12))
plt.imshow(corr)
plt.colorbar()
plt.xticks(label_loc, name_list, rotation=90)
plt.yticks(label_loc, name_list)
plt.tight_layout()
plt.savefig(f"{plot_dir}/corr.png")
plt.clf()
plt.close()




plt.figure(figsize=(28,8))

for my_spec in bin_out_dict.keys():
    id_spec, lb_spec = bin_out_dict[my_spec]
    plt.axvline(np.max(id_spec) + 0.5, color="black", alpha=0.4, linestyle="--")

plt.title(r"$\Sigma^{\rm comp}/\Sigma^{\rm tot}$", fontsize=22)
plt.plot(sub_x_ar_leakage_cov.diagonal()/sub_full_x_ar_cov.diagonal(), label="leakage cov / total cov")
plt.plot(sub_x_ar_beam_cov.diagonal()/sub_full_x_ar_cov.diagonal(), label="beam cov / total cov")
plt.plot(sub_x_ar_non_gaussian_cov_radio.diagonal()/sub_full_x_ar_cov.diagonal(), label="non gaussian radio cov / total cov")
plt.plot(sub_x_ar_non_gaussian_cov_lensing.diagonal()/sub_full_x_ar_cov.diagonal(), label="non gaussian lensing/ total cov")
plt.plot(sub_x_ar_non_gaussian_cov_tSZ.diagonal()/sub_full_x_ar_cov.diagonal(), label="non gaussian tSZ/ total cov")
plt.plot(sub_x_ar_non_gaussian_cov_CIB.diagonal()/sub_full_x_ar_cov.diagonal(), label="non gaussian CIB/ total cov")

plt.xticks(label_loc, name_list, rotation=90)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f"{plot_dir}/cov_different_component.png")
plt.clf()
plt.close()




for my_spec in bin_out_dict.keys():
    s_name, spectrum = my_spec
    id_spec, lb_spec = bin_out_dict[my_spec]
    plt.figure(figsize=(12, 8))
    plt.title(f"{s_name.replace('_', ' ')} {spectrum}", fontsize=22)
    plt.plot(lb_spec, sub_x_ar_cov.diagonal()[id_spec]/sub_full_x_ar_cov.diagonal()[id_spec], label="signal + noise")
    plt.plot(lb_spec, sub_x_ar_leakage_cov.diagonal()[id_spec]/sub_full_x_ar_cov.diagonal()[id_spec], label="T-> P leakage")
    plt.plot(lb_spec, sub_x_ar_beam_cov.diagonal()[id_spec]/sub_full_x_ar_cov.diagonal()[id_spec], label="beam")
    plt.plot(lb_spec, sub_x_ar_non_gaussian_cov_radio.diagonal()[id_spec]/sub_full_x_ar_cov.diagonal()[id_spec], label="connected trispectrum radio")
    plt.plot(lb_spec, sub_x_ar_non_gaussian_cov_lensing.diagonal()[id_spec]/sub_full_x_ar_cov.diagonal()[id_spec], label="connected trispectrum lensing")
    plt.plot(lb_spec, sub_x_ar_non_gaussian_cov_tSZ.diagonal()[id_spec]/sub_full_x_ar_cov.diagonal()[id_spec], label="connected trispectrum tSZ")
    plt.plot(lb_spec, sub_x_ar_non_gaussian_cov_CIB.diagonal()[id_spec]/sub_full_x_ar_cov.diagonal()[id_spec], label="connected trispectrum CIB")
    plt.legend(fontsize=18)
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.ylabel(r"$\Sigma^{\rm comp}_{\ell \ell}/\Sigma^{\rm tot}_{\ell \ell}$", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/cov_different_component_{s_name}_{spectrum}.png")
    plt.clf()
    plt.close()
