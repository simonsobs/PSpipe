"""
This script compute the amplitude of the BB power spectra of ACT DR6 using a dust prior from Planck 353
python results_BB_likelihood.py post_likelihood_PACT.dict
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, log, best_fits, external_data
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt
from cobaya.run import run
from matplotlib import rcParams
import matplotlib.ticker as ticker
import B_modes_utils

labelsize = 14 
fontsize = 20
divider_pow = -4

def loglike(a_BB_cmb, a_BB_dust):
    """ compute the loglike according to the parameters """

    theory = a_BB_cmb * vec_cmb_BB + a_BB_dust * vec_dust_BB_template
    residual = vec_BB - theory
    chi2 = residual @ i_cov_BB @ residual
    
    return -0.5 * chi2
    
def mcmc(mean_dust, std_dust, Rminus1_stop, Rminus1_cl_stop):

    """ the MCMC on the BB amplitude, mean_dust and std_dust are priors from Planck BB 353 GHz """

    info = {"likelihood": {"BB": loglike},
            "params": {
            "a_BB_cmb": {"prior": {"min": 0., "max": 2}, "latex": r"A_{BB}^{CMB}"},
            "a_BB_dust": {"prior": {"dist": "norm", "loc": mean_dust,"scale": std_dust}, "latex": r"A_{BB}^{dust}"}},
            "sampler": {"mcmc": {"max_tries": 10**6, "Rminus1_stop": Rminus1_stop, "Rminus1_cl_stop": Rminus1_cl_stop}},
            "resume": False,
            "output": f"{BB_dir}/mcmc/chain", "force": True}

    updated_info, sampler = run(info)

    samples = sampler.products(to_getdist=True, skip_samples=0.5)["sample"]
    
    return samples
    
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

cut = "post_unblinding"
Rminus1_stop = 0.01
Rminus1_cl_stop = 0.01

cov_dir = "covariances"
spec_dir = "spectra_leak_corr_ab_corr"
tag = d["best_fit_tag"]

BB_dir = f"plots/results_BB{tag}"
pspy_utils.create_directory(f"{BB_dir}/mcmc")

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]
cosmo_params = d["cosmo_params"]
accuracy_params = d["accuracy_params"]

fg_norm = d["fg_norm"]
fg_params = d["fg_params"]
fg_components = d["fg_components"]
do_bandpass_integration = d["do_bandpass_integration"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, delimiter="_", return_nu_tag=True)


passbands = {}
if do_bandpass_integration:
    log.info("Doing bandpass integration")

narrays, sv_list, ar_list = pspipe_list.get_arrays_list(d)
for sv, ar in zip(sv_list, ar_list):

    freq_info = d[f"freq_info_{sv}_{ar}"]
    if do_bandpass_integration:
        nu_ghz, pb = np.loadtxt(freq_info["passband"]).T
    else:
        nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.])
    passbands[f"{sv}_{ar}"] = [nu_ghz, pb]
    
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500, **accuracy_params)
fg_dict = best_fits.get_foreground_dict(l_th, passbands, fg_components, fg_params, fg_norm)

################################################################################################

spectra_cuts = B_modes_utils.get_spectra_cuts(cut, lmax)

if cut == "pre_unblinding":
    # list of new [bin_low, bin_max] for the final BB spectra
    #bin_scheme_edge = [[500, 750.5], [751, 1050.5], [1051, 1350.5], [1351, 1600.5], [1601, 2275.5], [2276, 9000]]
    bin_scheme_edge = [[500, 1200.5], [1201, 1500.5], [1501, 2000.5], [2001, 9000]]

if cut == "post_unblinding":
    bin_scheme_edge = [[500, 1200.5], [1201, 1500.5], [1501, 2000.5], [2001, 9000]]
   # bin_scheme_edge = [[500, 750.5], [751, 1050.5], [1051, 1350.5], [1351, 1600.5], [1601, 2275.5], [2276, 9000]]


only_TT_map_set = ["dr6_pa4_f220"]

bin_out_dict, indices = covariance.get_indices(bin_lo,
                                               bin_hi,
                                               lb,
                                               spec_name_list,
                                               spectra_cuts=spectra_cuts,
                                               spectra_order=spectra,
                                               selected_spectra=["BB"],
                                               only_TT_map_set=only_TT_map_set)
                                             
my_spectra = bin_out_dict.keys()

################################################################################################
cov_xar = np.load(f"{cov_dir}/x_ar_final_cov_data.npy")
cov_BB = cov_xar[np.ix_(indices, indices)]
i_cov_BB = np.linalg.inv(cov_BB)
corr_BB = so_cov.cov2corr(cov_BB, remove_diag=True)

plt.figure(figsize=(12,8))
plt.imshow(corr_BB)
plt.colorbar()
plt.savefig(f"{BB_dir}/full_correlation_{cut}.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()


vec_dust, vec_cmb = B_modes_utils.get_model_vecs(l_th, ps_dict, fg_dict, lmax, binning_file, spec_name_list, spectra)
vec_dust_BB = vec_dust[indices]
vec_cmb_BB = vec_cmb[indices]
vec_dust_BB_template = vec_dust_BB / fg_params["a_gbb"]

vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                           spec_name_list,
                                           "cross",
                                           spectra_order = spectra,
                                           type="Dl")

vec_BB = vec_xar[indices]




mean_dust, std_dust = fg_params["a_gbb"], 0.0084 #prior on data gal amplitude

samples = mcmc(mean_dust, std_dust, Rminus1_stop=Rminus1_stop, Rminus1_cl_stop=Rminus1_cl_stop)
gdplot = gdplt.get_subplot_plotter()
gdplot.triangle_plot(samples, ["a_BB_cmb", "a_BB_dust"], filled=True, title_limit=1)
plt.savefig(f"{paper_plot_dir}/posterior_BB_{cut}{tag}.pdf", bbox_inches="tight")
plt.clf()
plt.close()
    
#min_chi2 = np.min(samples.getParams().chi2)
chi2 = -2 * loglike(samples.mean("a_BB_cmb"), samples.mean("a_BB_dust"))

ndof = len(vec_BB) - 1
pte = 1 - ss.chi2(ndof).cdf(chi2)
print(f"{samples.mean('a_BB_cmb')}, {samples.std('a_BB_cmb')}, chi2: {chi2}, pte: {pte}, ndof: {ndof}")

lb_ml_BB, vec_ml_BB, cov_ml_BB = B_modes_utils.get_fg_sub_ML_solution_BB(lb,
                                                                         vec_BB,
                                                                         vec_dust_BB,
                                                                         i_cov_BB,
                                                                         bin_out_dict,
                                                                         bin_scheme_edge,
                                                                         fig_name=f"{BB_dir}/P_mat_all_BB.png")
std_ml_BB = np.sqrt(cov_ml_BB.diagonal())

plt.figure(figsize=(12,8))
plt.imshow(so_cov.cov2corr(cov_ml_BB, remove_diag=True))
plt.colorbar()
plt.savefig(f"{BB_dir}/ml_correlation_{cut}.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()
    
### Now do the plotting

add_BK = True
add_sptpol = True
add_polarbear = True
fac_ell = -1.
divider = 10**divider_pow
    
fig, ax = plt.subplots( figsize=(9, 5.5), dpi=100)
ax.plot(l_th, ps_dict["BB"] * l_th ** fac_ell / divider, color="gray")


ax.set_xlim(30, 4000)
if fac_ell == - 1:
    ax.set_ylim(-2*10**-1, 2.1)
if fac_ell == 0:
    ax.set_ylim(-0.05, 0.25)

ax.errorbar(lb_ml_BB, vec_ml_BB * lb_ml_BB ** fac_ell / divider, std_ml_BB * lb_ml_BB ** fac_ell / divider, fmt="o", color="royalblue", label="ACT", markersize=6, capsize=4, elinewidth=2)
if add_BK:
    ax.semilogx()
    lb_bicep, Db_bicep, err_bicep = external_data.get_bicep_BB_spectrum()
    ax.errorbar(lb_bicep, Db_bicep * lb_bicep ** fac_ell / divider, err_bicep * lb_bicep ** fac_ell / divider, fmt="o", color="red", label="BICEP/Keck (2021)", markersize=3, elinewidth=1)
if add_sptpol:
    lb_sptpol, Db_sptpol, err_sptpol = external_data.get_sptpol_BB_spectrum()
    ax.errorbar(lb_sptpol, Db_sptpol * lb_sptpol ** fac_ell / divider, err_sptpol * lb_sptpol ** fac_ell / divider, fmt="o", color="orange", label="SPTpol (2020)", markersize=3, elinewidth=1)
if add_polarbear:
    lb_polarbear, Db_polarbear, err_polarbear = external_data.get_polarbear_BB_spectrum()
    ax.errorbar(lb_polarbear, Db_polarbear * lb_polarbear ** fac_ell / divider, err_polarbear * lb_polarbear ** fac_ell / divider, fmt="o", color="green", label="POLARBEAR (2017)", markersize=3, elinewidth=1)

ax.plot(l_th, ps_dict["BB"] * 0, linestyle="--", color="black")

ax.legend(fontsize=labelsize)
ax.tick_params(labelsize=labelsize)

divider_str = r"10^{%s}" % divider_pow

plt.ylabel(r"$\ell^{-1} D_{\ell}^{BB} \ [{%s} \mu \rm K^{2}]$" % divider_str, fontsize=fontsize)
plt.xlabel(r"$\ell$", fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"{paper_plot_dir}/combined_BB_ellfac{fac_ell}_{cut}{tag}.pdf", bbox_inches="tight")
plt.clf()
plt.close()

np.savetxt(f"{BB_dir}/combined_BB.dat", np.transpose([lb_ml_BB, vec_ml_BB, std_ml_BB]))
    
# also plot individual spectra for the data
for my_spec in bin_out_dict.keys():
    s_name, spectrum = my_spec
    id, lb = bin_out_dict[my_spec]
    lb_, Db = so_spectra.read_ps(f"{spec_dir}/Dl_{s_name}_cross.dat", spectra=spectra)
        
    plt.figure(figsize=(12,8))
    plt.title(f"{my_spec}, min={np.min(lb)}, max={np.max(lb)}")
    plt.plot(lb_, Db[spectrum], label="original spectrum")
    plt.errorbar(lb, vec_BB[id], np.sqrt(cov_BB[np.ix_(id,id)].diagonal()), fmt=".", label="selected spectrum")
    plt.plot(lb, vec_dust_BB[id] + vec_cmb_BB[id], "--", color="gray",alpha=0.3, label="theory")
    plt.ylim(-1, 1)
    plt.legend()
    plt.savefig(f"{BB_dir}/{spectrum}_{s_name}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
        
    res = vec_BB[id] - vec_dust_BB[id] - vec_cmb_BB[id]
    chi2 = res @ np.linalg.inv(cov_BB[np.ix_(id, id)]) @ res
    ndof = len(lb) - 1
    pte = 1 - ss.chi2(ndof).cdf(chi2)
    print(my_spec, f"chi2: {chi2}, ndof: {ndof}, pte: {pte}")
