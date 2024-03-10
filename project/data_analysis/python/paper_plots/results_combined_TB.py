"""
This script combine all cross TB and BT spectra into a final TB power spectrum
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
spec_dir = "spectra_leak_corr_polangle_corr"
result_dir = "result_TB"
plot_dir = "plots/combined_cov"
best_fit_dir = "best_fits"
pspy_utils.create_directory(result_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
lmin = 475

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, delimiter="_", return_nu_tag=True)

vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                           spec_name_list,
                                           "cross",
                                           spectra_order = spectra,
                                           type="Dl")

cov_xar = np.load(f"{cov_dir}/x_ar_final_cov_data.npy")

################################################################################################
# Start at l=500, remove pa4_f220 pol and only include TB/BT

selected_spectra = ["TB", "BT"]
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[lmax, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa6_f150": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa5_f090": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa6_f090": dict(T=[lmin, lmax], P=[lmin, lmax]),
}

only_TT_map_set = ["dr6_pa4_f220"]

bin_out_dict, indices = covariance.get_indices(bin_lo,
                                               bin_hi,
                                               lb,
                                               spec_name_list,
                                               spectra_cuts=spectra_cuts,
                                               spectra_order=spectra,
                                               selected_spectra=selected_spectra,
                                               only_TT_map_set=only_TT_map_set)
                                             
my_spectra = bin_out_dict.keys()
################################################################################################


id = np.where(bin_lo>lmin)
bin_lo, bin_hi, lb, bin_size = bin_lo[id], bin_hi[id], lb[id], bin_size[id]
n_bins = len(bin_hi)

cov_TB = cov_xar[np.ix_(indices, indices)]
vec_TB = vec_xar[indices]

# create the passage matrix combining all spectra
n_spec = len(my_spectra)
P_mat = np.zeros((n_spec * n_bins, n_bins))
name_list = []
for i_spec, my_spec in enumerate(my_spectra):
    P_mat[i_spec *  n_bins: (i_spec + 1) *  n_bins, :] = np.identity(n_bins)
    
    name, spectrum = my_spec
    name = name.replace("dr6_", "")
    name_list += [f"{spectrum} {name}"]
    

# plot the TB/BT correlation matrix
plt.figure(figsize=(16, 16))
plt.imshow(so_cov.cov2corr(cov_TB))
plt.xticks(ticks=np.arange(n_spec) * n_bins + n_bins/2, labels = name_list, rotation=90, fontsize=20)
plt.yticks(ticks=np.arange(n_spec) * n_bins + n_bins/2, labels = name_list, fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{result_dir}/correlation_TB.png")
#plt.show()
plt.clf()
plt.close()


# do the max likelihood combination of all spectra

i_cov_TB = np.linalg.inv(cov_TB)
chi2 = vec_TB @ i_cov_TB @  vec_TB
ndof = len(vec_TB)
pte = 1 - ss.chi2(ndof).cdf(chi2)
print(chi2, pte)


cov_TB_ML = covariance.get_max_likelihood_cov(P_mat,
                                              i_cov_TB,
                                              force_sim = True,
                                              check_pos_def = True)


vec_ML = covariance.max_likelihood_spectra(cov_TB_ML,
                                           i_cov_TB,
                                           P_mat,
                                           vec_TB)

# compute the chi2 and PTE

i_cov_TB_ML = np.linalg.inv(cov_TB_ML)
chi2 = vec_ML @ i_cov_TB_ML @  vec_ML
ndof = len(vec_ML)
pte = 1 - ss.chi2(ndof).cdf(chi2)

lth, psth = so_spectra.read_ps(f"{best_fit_dir}/cmb.dat", spectra=spectra)

std = np.sqrt(cov_TB_ML.diagonal())

plt.figure(figsize=(16, 16))
plt.imshow(so_cov.cov2corr(cov_TB_ML, remove_diag=True))
plt.xticks(ticks=np.arange(len(lb)), labels = lb, rotation=90)
plt.yticks(ticks=np.arange(len(lb)), labels = lb)
plt.colorbar()
plt.plt.savefig(f"{result_dir}/cov_TB_ML.png")
plt.clf()
plt.close()


plt.figure(figsize=(12, 8))
plt.errorbar(lb, vec_ML, std, fmt="o", label=r"$\chi^{2}$=%.02f, $n_{DoF}$ = %d, PTE = %.02f" % (chi2, ndof, pte), color="red")
plt.plot(lth, psth["TE"]/100, label=r"$D_{\ell}^{TE}$/100", linestyle="--", alpha=0.7, color="navy")
plt.plot(lth, lth*0, color="gray")
plt.ylabel(r"$D_{\ell}^{TB}$", fontsize=22)
plt.xlabel(r"$\ell$", fontsize=22)
plt.legend(fontsize=18)
plt.xlim(0,8000)
plt.plt.savefig(f"{result_dir}/combined_TB.png")
plt.clf()
plt.close()

###### this is a test with the old way of doing thing (don't delete this is a useful test)

#bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
#n_bins = len(lb)
#id = np.where(bin_lo>lmin)

#vec_test = []
#err_test = []
#for my_spec in my_spectra:
#    name, spectrum = my_spec
#    lb, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{name}_cross.dat", spectra=spectra)
#    vec_test = np.append(vec_test, ps[spectrum][id])
#    mc_cov = np.load(f"{cov_dir}/mc_cov_{name}_{name}.npy")
#    beam_cov = np.load(f"{cov_dir}/beam_cov_{name}_{name}.npy")
#    leakage_cov = np.load(f"{cov_dir}/leakage_cov_{name}_{name}.npy")
#    mc_cov += beam_cov + leakage_cov
#    mc_cov = so_cov.selectblock(mc_cov, spectra, n_bins, block=spectrum+spectrum)
#    err_test = np.append(err_test, np.sqrt(mc_cov.diagonal()[id]))


#bins = np.arange(len(vec_TB))
#plt.errorbar(bins, vec_TB)
#plt.errorbar(bins, vec_test, fmt= "--")
#plt.show()

#plt.errorbar(bins, np.sqrt(cov_TB.diagonal()))
#plt.errorbar(bins, err_test, fmt= "--")
#plt.show()
