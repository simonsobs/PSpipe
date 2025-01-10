"""
This script compute the amplitude of the BB power spectra of ACT DR6 simulation
python mc_BB_likelihood.py global_dr6_v4.dict
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

rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

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

cut = "pre_unblinding"
Rminus1_stop = 0.03
Rminus1_cl_stop = 0.03

cov_dir = "covariances"
sim_spec_dir = d["sim_spec_dir"]

BB_dir = f"results_BB_MC"

pspy_utils.create_directory(f"{BB_dir}/mcmc")

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
    bin_scheme_edge = [[500, 750.5], [751, 1050.5], [1051, 1350.5], [1351, 1600.5], [1601, 2275.5], [2276, 9000]]
if cut == "post_unblinding":
    bin_scheme_edge = [[500, 1200], [1201, 1500], [1501, 2000], [2001, 9000]]


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

mean_dust, std_dust = fg_params["a_gbb"], 0.0084 #prior on data gal amplitude



vec_ml_list = []
chi2_list, a_cmb_list, a_dust_list = [], [], []
test_sampling = True
n_sims = 100
for iii in range(n_sims):
    print(iii)
    vec_xar = covariance.read_x_ar_spectra_vec(sim_spec_dir,
                                               spec_name_list,
                                               f"cross_{iii:05d}",
                                               spectra_order = spectra,
                                               type="Dl")

    vec_BB = vec_xar[indices]

    lb_ml_BB, vec_ml_BB, cov_ml_BB = B_modes_utils.get_fg_sub_ML_solution_BB(lb,
                                                                             vec_BB,
                                                                             vec_dust_BB,
                                                                             i_cov_BB,
                                                                             bin_out_dict,
                                                                             bin_scheme_edge)
    std_ml_BB = np.sqrt(cov_ml_BB.diagonal())

    vec_ml_list += [vec_ml_BB]

    if test_sampling == True:
        samples = mcmc(mean_dust, std_dust, Rminus1_stop=Rminus1_stop, Rminus1_cl_stop=Rminus1_cl_stop)
        chi2 = -2 * loglike(samples.mean("a_BB_cmb"), samples.mean("a_BB_dust"))
        ndof = len(vec_BB)
        print(f"chi2: {chi2}, ndof: {ndof}")

        chi2_list += [chi2]
        a_cmb_list += [samples.mean("a_BB_cmb")]
        a_dust_list += [samples.mean("a_BB_dust")]


mean_vec, std_vec = np.mean(vec_ml_list, axis=0), np.std(vec_ml_list, axis=0)
cov_mc = np.cov(vec_ml_list, rowvar=False)
    
plt.figure(figsize=(12,8))
plt.plot(lb_ml_BB, np.sqrt(cov_ml_BB.diagonal()))
plt.plot(lb_ml_BB, std_vec)
plt.savefig(f"{BB_dir}/error_comp_{cut}.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()

plt.figure(figsize=(12,8))
plt.imshow(so_cov.cov2corr(cov_mc, remove_diag=True))
plt.colorbar()
plt.savefig(f"{BB_dir}/mc_correlation_{cut}.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()

plt.figure(figsize=(12,8))
plt.ylim(-0.1, 0.25)
plt.xlim(0, 4000)
plt.plot(l_th, ps_dict["BB"])
plt.errorbar(lb_ml_BB, mean_vec, std_vec)
plt.ylabel(r"$D_{\ell}^{BB}$", fontsize=22)
plt.xlabel(r"$\ell$", fontsize=22)
plt.savefig(f"{BB_dir}/combined_BB_sim_{cut}.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()

if test_sampling == True:
    mean, std = np.mean(a_cmb_list, axis=0), np.std(a_cmb_list, axis=0)
    print("mean", mean, "std", std, "std MC", np.sqrt(samples.cov(["a_BB_cmb"])[0, 0]), std/np.sqrt(n_sims))
    mean_chi2 = np.mean(chi2, axis=0)
    print("mean chi2", mean_chi2)
