"""
This script compute the pol angles using all EB/BE spectra
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, pol_angle
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from cobaya.run import run
from getdist import plots
import matplotlib as mpl

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
spec_dir = "spectra_leak_corr"
result_dir = "pol_angle"
bestfit_dir = "best_fits"

pspy_utils.create_directory(result_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
lmin = 475

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
id = np.where(bin_lo > lmin)

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, delimiter="_", return_nu_tag=True)


vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                           spec_name_list,
                                           "cross",
                                           spectra_order = spectra,
                                           type="Dl")

cov_xar = np.load(f"{cov_dir}/x_ar_final_cov_data.npy")


################################################################################################

# Start at l=500, remove pa4_f220 pol and only include EB/BE
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[lmax, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa6_f150": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa5_f090": dict(T=[lmin, lmax], P=[lmin, lmax]),
    "dr6_pa6_f090": dict(T=[lmin, lmax], P=[lmin, lmax]),
}

bin_out_dict, indices = covariance.get_indices(bin_lo,
                                               bin_hi,
                                               lb,
                                               spec_name_list,
                                               spectra_cuts=spectra_cuts,
                                               spectra_order=spectra,
                                               selected_spectra=["EB", "BE"])
                                               
my_spectra = bin_out_dict.keys()

################################################################################################
cov_EB = cov_xar[np.ix_(indices, indices)]
vec_EB = vec_xar[indices]


n_spec = len(my_spectra)
n_bins = len(lb[id])
name_list = []
for my_spec in my_spectra:
    name, spectrum = my_spec
    name = name.replace("dr6_", "")
    name_list += [f"{spectrum} {name}"]

# plot the TB/BT correlation matrix
plt.figure(figsize=(16, 16))
plt.imshow(so_cov.cov2corr(cov_EB))
plt.xticks(ticks=np.arange(n_spec) * n_bins + n_bins/2, labels = name_list, rotation=90, fontsize=20)
plt.yticks(ticks=np.arange(n_spec) * n_bins + n_bins/2, labels = name_list, fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{result_dir}/correlation_EB.png")
#plt.show()
plt.clf()
plt.close()




i_cov = np.linalg.inv(cov_EB)

lth, psth = so_spectra.read_ps(bestfit_dir + f"/cmb.dat", spectra=spectra)
lb_, psth_b = so_spectra.bin_spectra(lth,
                                     psth,
                                     d["binning_file"],
                                     d["lmax"],
                                     type="Cl",
                                     spectra=spectra)

def get_vec_th_EB(alpha_pa5_f090, alpha_pa5_f150, alpha_pa6_f090, alpha_pa6_f150):

    alpha = {}
    alpha["dr6_pa5_f090"] = alpha_pa5_f090
    alpha["dr6_pa5_f150"] = alpha_pa5_f150
    alpha["dr6_pa6_f090"] = alpha_pa6_f090
    alpha["dr6_pa6_f150"] = alpha_pa6_f150

    vec_th_EB = []
    for spec in my_spectra:
        spec_name, mode = spec

        n1, n2 = spec_name.split("x")
        _, psth_rot = pol_angle.rot_theory_spectrum(lb, psth_b, alpha[n1], alpha[n2])
        vec_th_EB = np.append(vec_th_EB, psth_rot[mode][id])
        
    return vec_th_EB


def loglike(alpha_pa5_f090, alpha_pa5_f150, alpha_pa6_f090, alpha_pa6_f150):
    
    vec_th_EB = get_vec_th_EB(alpha_pa5_f090, alpha_pa5_f150, alpha_pa6_f090, alpha_pa6_f150)
    res = vec_EB - vec_th_EB
    chi2 =  res @ i_cov @ res

    return -0.5 * chi2
    
    
roots = ["mcmc"]
info = {}
info["likelihood"] = { "my_like": loglike}
info["params"] = { "alpha_pa5_f090": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa5 f090}"},
                   "alpha_pa5_f150": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa5 f150}"},
                   "alpha_pa6_f090": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa6 f090}"},
                   "alpha_pa6_f150": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa6 f150}"}}
info["sampler"] = {  "mcmc": {  "max_tries": 1e6,  "Rminus1_stop": 0.02, "Rminus1_cl_stop": 0.04}}
info["output"] = f"{result_dir}/chains/{roots[0]}"
info["force"] = True
info["debug"] = False
updated_info, sampler = run(info)

params = ["alpha_pa5_f090", "alpha_pa5_f150", "alpha_pa6_f090", "alpha_pa6_f150"]

g = plots.get_subplot_plotter(
    chain_dir=os.path.join(os.getcwd(), f"{result_dir}/chains"),
    analysis_settings={"ignore_rows": 0.5},
)
kwargs = dict(colors=["k"], lws=[1])
g.triangle_plot(roots, params, **kwargs, diag1d_kwargs=kwargs)

# Add table on figure

with mpl.rc_context(rc={"text.usetex": True}):
    table = g.sample_analyser.mcsamples[roots[0]].getTable(limit=1, paramList=params)
    kwargs = dict(size=15, ha="right")
    g.subplots[0, 0].text(1.2, 0,  table.tableTex().replace("\n", ""), **kwargs)

plt.savefig(f"{result_dir}/pol_angles_chain_results.pdf")
plt.clf()
plt.close()

samples = sampler.products(to_getdist=True, skip_samples=0.5)["sample"]
min_chi2 = np.min(samples.getParams().chi2)

ndof = len(vec_EB) - 4
pte = 1 - ss.chi2(ndof).cdf(min_chi2)
print(f"min chi2 = {min_chi2}, pte = {pte}")




alpha_pa5_f090 = 0.097
alpha_pa5_f150 = 0.356
alpha_pa6_f090 = 0.179
alpha_pa6_f150 = 0.220

nbin_tot = len(vec_EB)
bin = np.arange(nbin_tot)
error_EB = np.sqrt(np.diagonal(cov_EB))
vec_EB_corr = vec_EB - get_vec_th_EB(alpha_pa5_f090, alpha_pa5_f150, alpha_pa6_f090, alpha_pa6_f150)


chi2_precorr =  vec_EB @ i_cov @ vec_EB
chi2_postcorr =  vec_EB_corr @ i_cov @ vec_EB_corr

PTE_precorr  = 1 - ss.chi2(nbin_tot).cdf(chi2_precorr)
PTE_postcorr  = 1 - ss.chi2(nbin_tot - 4).cdf(chi2_postcorr)


plt.figure(figsize=(16,8))
plt.ylabel(r"$D_{\ell}^{EB}/\sigma_{\ell}^{EB}$", fontsize=22)
plt.plot(bin, vec_EB / error_EB, label=r"pre-correction $\chi^{2}$=%.02f, $n_{DoF}$ = %d, PTE = %.04f" % (chi2_precorr, nbin_tot, PTE_precorr), color="gray")
plt.plot(bin, vec_EB_corr / error_EB, label=r"post-correction  $\chi^{2}$=%.02f, $n_{DoF}$ = %d, PTE = %.04f" % (chi2_postcorr, nbin_tot - 4, PTE_postcorr), color="orange", alpha=0.6)
plt.plot(bin, bin*0, "--", color="gray")
plt.xticks(ticks=np.arange(n_spec) * n_bins + n_bins/2, labels = name_list, rotation=90, fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f"{result_dir}/vec_EB.png")
#plt.show()
plt.clf()
plt.close()
