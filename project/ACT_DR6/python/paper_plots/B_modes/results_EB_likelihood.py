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
from getdist import plots, loadMCSamples, MCSamples
import matplotlib as mpl
from matplotlib import cm
import B_modes_utils

def get_vec_th_EB(alpha_pa5_f090, alpha_pa5_f150, alpha_pa6_f090, alpha_pa6_f150):

    alpha = {}
    alpha["dr6_pa5_f090"] = alpha_pa5_f090
    alpha["dr6_pa5_f150"] = alpha_pa5_f150
    alpha["dr6_pa6_f090"] = alpha_pa6_f090
    alpha["dr6_pa6_f150"] = alpha_pa6_f150
    vec_th_EB = []
    for spec in my_spectra:
        spec_name, mode = spec
        id_spec, lb_spec = bin_out_dict[spec]
        id = np.where(lb >= lb_spec[0])

        n1, n2 = spec_name.split("x")
        _, psth_rot = pol_angle.rot_theory_spectrum(lb, psth_b, alpha[n1], alpha[n2])
        vec_th_EB = np.append(vec_th_EB, psth_rot[mode][id])
        
    return vec_th_EB

def loglike(alpha_pa5_f090, alpha_pa5_f150, alpha_pa6_f090, alpha_pa6_f150):

    vec_th_EB = get_vec_th_EB(alpha_pa5_f090, alpha_pa5_f150, alpha_pa6_f090, alpha_pa6_f150)
    res = vec_EB - vec_th_EB
    chi2 =  res @ i_cov @ res
    return -0.5 * chi2
    

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cut = "post_unblinding"
Rminus1_stop = 0.01
Rminus1_cl_stop = 0.02

cov_dir = "covariances"
spec_dir = "spectra_leak_corr_ab_corr"
tag = d["best_fit_tag"]

bestfit_dir = f"best_fits{tag}"


paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

result_dir = f"plots/results_EB{tag}"
pspy_utils.create_directory(result_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, delimiter="_", return_nu_tag=True)

################################################################################################
# Start at l=500, remove pa4_f220 pol and only include EB/BE

spectra_cuts = B_modes_utils.get_spectra_cuts(cut, lmax)

only_TT_map_set = ["dr6_pa4_f220"]
bin_out_dict, indices = covariance.get_indices(bin_lo,
                                               bin_hi,
                                               lb,
                                               spec_name_list,
                                               spectra_cuts=spectra_cuts,
                                               spectra_order=spectra,
                                               selected_spectra=["EB", "BE"],
                                               only_TT_map_set=only_TT_map_set)

my_spectra = bin_out_dict.keys()
################################################################################################
cov_xar = np.load(f"{cov_dir}/x_ar_final_cov_data.npy")
vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                           spec_name_list,
                                           "cross",
                                           spectra_order = spectra,
                                           type="Dl")


cov_EB = cov_xar[np.ix_(indices, indices)]
vec_EB = vec_xar[indices]
i_cov = np.linalg.inv(cov_EB)

corr_EB = so_cov.cov2corr(cov_EB, remove_diag=True)

tick_start = 0
tick_loc_list, name_list = [], []
for spec in my_spectra:
    name, spectrum = spec
    id_spec, lb_spec = bin_out_dict[spec]
    tick_loc_list += [len(lb_spec)/2 + tick_start]
    tick_start += len(lb_spec)
    name = name.replace("dr6_", "")
    name_list += [f"{spectrum} {name}"]

# plot the EB/BE correlation matrix
plt.figure(figsize=(16, 16))
plt.imshow(corr_EB)
plt.xticks(ticks=tick_loc_list, labels = name_list, rotation=90, fontsize=20)
plt.yticks(ticks=tick_loc_list, labels = name_list, fontsize=20)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{result_dir}/correlation_EB.png")
plt.clf()
plt.close()

################################################################################################
# Run a MCMC chain and plot the posterior distribution

lth, psth = so_spectra.read_ps(bestfit_dir + f"/cmb.dat", spectra=spectra)
_, psth_b = so_spectra.bin_spectra(lth,
                                   psth,
                                   d["binning_file"],
                                   d["lmax"],
                                   type="Cl",
                                   spectra=spectra)


sample = True
roots = ["mcmc"]
params = ["alpha_pa5_f090", "alpha_pa5_f150", "alpha_pa6_f090", "alpha_pa6_f150"]

if sample:
    info = {}
    info["likelihood"] = { "my_like": loglike}
    info["params"] = { "alpha_pa5_f090": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa5 f090}"},
                       "alpha_pa5_f150": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa5 f150}"},
                       "alpha_pa6_f090": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa6 f090}"},
                       "alpha_pa6_f150": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": r"\alpha_{pa6 f150}"}}
    info["sampler"] = {"mcmc": {  "max_tries": 1e6,  "Rminus1_stop": Rminus1_stop, "Rminus1_cl_stop": Rminus1_cl_stop}}
    info["output"] = f"{result_dir}/chains_{cut}/{roots[0]}"
    info["force"] = True
    info["debug"] = False
    updated_info, sampler = run(info)

burnin = 0.5
g = plots.get_subplot_plotter(
    chain_dir=os.path.join(os.getcwd(), f"{result_dir}/chains_{cut}"),
    analysis_settings={"ignore_rows": burnin},
)
kwargs = dict(colors=["k"], lws=[1])
g.triangle_plot(roots, params, **kwargs, diag1d_kwargs=kwargs)

# Add table on figure
with mpl.rc_context(rc={"text.usetex": True}):
    table = g.sample_analyser.mcsamples[roots[0]].getTable(limit=1, paramList=params)
    kwargs = dict(size=15, ha="right")
    g.subplots[0, 0].text(1.2, 0.2,  table.tableTex().replace("\n", ""), **kwargs)

plt.savefig(f"{paper_plot_dir}/pol_angles_chain_results_{cut}{tag}.pdf")
plt.clf()
plt.close()

samples = loadMCSamples(f"{result_dir}/chains_{cut}/{roots[0]}", settings={"ignore_rows": burnin})


nbin_tot = len(vec_EB)
bin = np.arange(nbin_tot)
error_EB = np.sqrt(np.diagonal(cov_EB))
vec_EB_corr = vec_EB - get_vec_th_EB(samples.mean("alpha_pa5_f090"),
                                     samples.mean("alpha_pa5_f150"),
                                     samples.mean("alpha_pa6_f090"),
                                     samples.mean("alpha_pa6_f150"))
                                     

chi2_precorr =  vec_EB @ i_cov @ vec_EB
chi2_postcorr =  vec_EB_corr @ i_cov @ vec_EB_corr

min_chi2 = np.min(samples.getParams().chi2)
ndof = len(vec_EB) - 4
pte = 1 - ss.chi2(ndof).cdf(min_chi2)
print(f"min chi2 = {min_chi2}, pte = {pte}")

PTE_precorr  = 1 - ss.chi2(nbin_tot).cdf(chi2_precorr)
PTE_postcorr  = 1 - ss.chi2(nbin_tot - 4).cdf(chi2_postcorr)

plt.figure(figsize=(16,8))
plt.ylabel(r"$D_{\ell}^{EB}/\sigma_{\ell}^{EB}$", fontsize=22)
plt.plot(bin, vec_EB / error_EB, label=r"pre-correction $\chi^{2}$=%.02f, $n_{DoF}$ = %d, PTE = %.04f" % (chi2_precorr, nbin_tot, PTE_precorr), color="gray")
plt.plot(bin, vec_EB_corr / error_EB, label=r"post-correction  $\chi^{2}$=%.02f, $n_{DoF}$ = %d, PTE = %.04f" % (chi2_postcorr, nbin_tot - 4, PTE_postcorr), color="orange", alpha=0.6)
plt.plot(bin, bin*0, "--", color="gray")
plt.xticks(ticks=tick_loc_list, labels = name_list, rotation=90, fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f"{paper_plot_dir}/vec_EB_{cut}{tag}.pdf")
plt.clf()
plt.close()


lb_ml = B_modes_utils.get_ml_bins(bin_out_dict, lb)
P_mat = B_modes_utils.get_P_mat(len(vec_EB), lb_ml, bin_out_dict, fig_name=f"{result_dir}/P_mat_all_EB{cut}.png")

cov_ml = covariance.get_max_likelihood_cov(P_mat,
                                           i_cov,
                                           force_sim = True,
                                           check_pos_def = True)
                                           
vec_ml = covariance.max_likelihood_spectra(cov_ml,
                                           i_cov,
                                           P_mat,
                                           vec_EB)


error_ml = np.sqrt(cov_ml.diagonal())

np.savetxt(f"{result_dir}/combined_EB_{cut}.dat", np.transpose([lb_ml, vec_ml, error_ml]))
np.save(f"{result_dir}/combined_cov_EB_{cut}.npy", cov_ml)


