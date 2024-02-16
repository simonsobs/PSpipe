"""
This script compute the amplitude of the BB power spectra of ACT DR6 using a dust prior from Planck 353
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
import scipy.stats as ss
from matplotlib import rcParams


rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20


def get_and_select_data_vec(spec_dir,
                            spec_tag,
                            spec_name_list,
                            spectra,
                            type,
                            indices):
    """ get the selected data vector  """
    
    vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                               spec_name_list,
                                               spec_tag,
                                               spectra_order = spectra,
                                               type=type)

    sub_vec = vec_xar[indices]
    
    return sub_vec
    
def get_and_select_model_vecs(cosmo_params,
                              accuracy_params,
                              fg_components,
                              fg_params,
                              fg_norm,
                              passbands,
                              type,
                              lmax,
                              binning_file,
                              spec_name_list,
                              spectra,
                              indices):
                              
    """ get the selected model vector for CMB and fg  """
    
    log.info(f"Computing CMB spectra with cosmological parameters: {cosmo_params}")
    l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500, **accuracy_params)

    log.info("Getting foregrounds contribution")

    fg_dict = best_fits.get_foreground_dict(l_th, passbands, fg_components, fg_params, fg_norm)

    vec_cmb, vec_fg = [],  []
    plt.figure(figsize=(12,10))
    for spec in spectra:
        lb, ps_b =  pspy_utils.naive_binning(l_th, ps_dict[spec], binning_file, lmax)
        
        if spec == "BB":
            plt.plot(l_th, ps_dict["BB"], color="gray")

        for spec_name in spec_name_list:
            na, nb = spec_name.split("x")
            fg = fg_dict["bb", "all", na, nb]
            lb, fg_b = pspy_utils.naive_binning(l_th, fg, binning_file, lmax)
            if (spec == "ET" or spec == "BT" or spec == "BE") & (na == nb): continue
            vec_fg = np.append(vec_fg, fg_b)
            vec_cmb = np.append(vec_cmb, ps_b)
            
            if (spec == "BB") & ("pa4" not in spec_name):
                plt.plot(l_th, fg, label = spec_name)
    plt.ylim(0, 0.15)
    plt.xlim(0, 4000)
    plt.ylabel(r"$D_{\ell}^{BB}$", fontsize=22)
    plt.xlabel(r"$\ell$", fontsize=22)
    plt.legend(fontsize=12)
    plt.savefig(f"{BB_dir}/foreground.png", bbox_inches="tight")
    plt.clf()
    plt.close()

        
    vec_fg_BB = vec_fg[indices]
    vec_cmb_BB = vec_cmb[indices]
    
    return vec_fg_BB, vec_cmb_BB
    
def get_ML_solution(vec_data_BB, vec_fg_BB, i_cov_BB, n_spec, n_bins, meta_bin_scheme):
    """ this project all BB spectra into one with a sparser binning scheme defined by meta_bin_scheme"""

    def get_P_mat(n_spec, n_bins,  meta_bin_scheme):

        new_n_bins = len(meta_bin_scheme)
        sub_P_mat = np.zeros((n_bins, new_n_bins))
        for i in range(new_n_bins):
            if i == 0:
                not_zero_loc = np.arange(meta_bin_scheme[i])
            else:
                not_zero_loc = np.arange(meta_bin_scheme[i]) + np.max(not_zero_loc) + 1
            sub_P_mat[not_zero_loc, i] = 1

        full_P_mat = np.zeros((n_bins * n_spec, new_n_bins))
        for i_spec in range(n_spec):
            full_P_mat[i_spec * n_bins: (i_spec + 1) * n_bins, :] = sub_P_mat
            
        return full_P_mat
        
    full_P_mat = get_P_mat(n_spec, n_bins,  meta_bin_scheme)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(full_P_mat)
    plt.gca().set_aspect(0.02)
    plt.savefig(f"{BB_dir}/P_mat.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()

    cov_ML_BB = covariance.get_max_likelihood_cov(full_P_mat,
                                                  i_cov_BB,
                                                  force_sim = True,
                                                  check_pos_def = True)
                                                  
    plt.figure(figsize=(12,8))
    plt.imshow(so_cov.cov2corr(cov_ML_BB, remove_diag=True))
    plt.colorbar()
    plt.savefig(f"{BB_dir}/correlation.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()

    vec_ML_BB = covariance.max_likelihood_spectra(cov_ML_BB,
                                                 i_cov_BB,
                                                 full_P_mat,
                                                 vec_data_BB - vec_fg_BB)

    lb_vec = np.tile(lb, n_spec)
    lb_ML_BB = covariance.max_likelihood_spectra(cov_ML_BB,
                                                 i_cov_BB,
                                                 full_P_mat,
                                                 lb_vec)

    return lb_ML_BB, vec_ML_BB, cov_ML_BB
    
def loglike(a_BB_cmb, a_BB_dust):
    """ compute the loglike according to the parameters """

    theory = a_BB_cmb * vec_cmb_BB + a_BB_dust * vec_fg_BB_template
    residual = vec_data_BB - theory
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

cov_dir = "covariances"
spec_dir = "spectra_leak_corr"
sim_spec_dir = "sim_spectra"
BB_dir = "results_BB"

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
lmin = 475

l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500, **accuracy_params)

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

################################################################################################
# Start at l=500, remove pa4_f220 pol and only include TB/BT

selected_spectra = ["BB"]
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

n_spec = len(my_spectra)

id = np.where(bin_lo>lmin)
bin_lo, bin_hi, lb, bin_size = bin_lo[id], bin_hi[id], lb[id], bin_size[id]
n_bins = len(bin_hi)
print(n_bins)


cov_xar = np.load(f"{cov_dir}/x_ar_final_cov_data.npy")

cov_BB = cov_xar[np.ix_(indices, indices)]

plt.figure(figsize=(12,8))
plt.imshow(so_cov.cov2corr(cov_BB, remove_diag=True))
plt.colorbar()
plt.savefig(f"{BB_dir}/full_correlation.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()

i_cov_BB = np.linalg.inv(cov_BB)

vec_fg_BB, vec_cmb_BB = get_and_select_model_vecs(cosmo_params,
                                                  accuracy_params,
                                                  fg_components,
                                                  fg_params,
                                                  fg_norm,
                                                  passbands,
                                                  type,
                                                  lmax,
                                                  binning_file,
                                                  spec_name_list,
                                                  spectra,
                                                  indices)

# create a fg template that A_BB_dust multiply
vec_fg_BB_template = vec_fg_BB / fg_params["a_gbb"]

# we will rebin according the this scheme
#(the entry of the list correspond to the number of old bin

meta_bin_scheme = [3, 3, 3, 3, 3, 4, 4, 11, 15]
print(np.sum(meta_bin_scheme), n_bins)
assert(np.sum(meta_bin_scheme)  == n_bins)

sim = False
if sim == True:
    n_sims = 100
    mean_dust, std_dust = 0.114, 0.0084 #prior on sim gal amplitude

    a_BB_cmb_list, vec_ml_BB_list = [], []
    cov_mc = 0

    for iii in range(n_sims):
        vec_data_BB  = get_and_select_data_vec(sim_spec_dir, f"cross_{iii:05d}", spec_name_list, spectra, type, indices)
        samples = mcmc(mean_dust, std_dust, Rminus1_stop=0.03, Rminus1_cl_stop=0.03)
        
        a_BB_cmb_list += [samples.mean("a_BB_cmb")]
        
        lb_ml_BB, vec_ml_BB, cov_ml_BB = get_ML_solution(vec_data_BB, vec_fg_BB, i_cov_BB, n_spec, n_bins, meta_bin_scheme)
        vec_ml_BB_list += [vec_ml_BB]
        
        if iii == 0:
            gdplot = gdplt.get_subplot_plotter()
            gdplot.triangle_plot(samples, ["a_BB_cmb", "a_BB_dust"], filled=True, title_limit=1)
            plt.savefig(f"{BB_dir}/posterior_sim_BB.png", dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()
        
        
        cov_mc += np.outer(vec_ml_BB, vec_ml_BB)

    mean, std = np.mean(a_BB_cmb_list, axis=0), np.std(a_BB_cmb_list, axis=0)
    mean_vec, std_vec = np.mean(vec_ml_BB_list, axis=0), np.std(vec_ml_BB_list, axis=0)
    
    print("mean", mean, "std", std, "std MC", np.sqrt(samples.cov(["a_BB_cmb"])[0, 0]), std/np.sqrt(n_sims))
    
    
    cov_mc = cov_mc/n_sims -  np.outer(mean_vec, mean_vec)
    
    plt.figure(figsize=(12,8))
    plt.plot(lb_ml_BB, np.sqrt(cov_ml_BB.diagonal()))
    plt.plot(lb_ml_BB, std_vec)
    plt.savefig(f"{BB_dir}/error_comp.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure(figsize=(12,8))
    plt.imshow(so_cov.cov2corr(cov_mc, remove_diag=True))
    plt.colorbar()
    plt.savefig(f"{BB_dir}/mc_correlation.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure(figsize=(12,8))
    plt.ylim(-0.1, 0.25)
    plt.xlim(0, 4000)
    plt.plot(l_th, ps_dict["BB"])
    plt.errorbar(lb_ml_BB, mean_vec, std_vec)
    plt.ylabel(r"$D_{\ell}^{BB}$", fontsize=22)
    plt.xlabel(r"$\ell$", fontsize=22)
    plt.savefig(f"{BB_dir}/combined_BB_sim.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
    
else:
    mean_dust, std_dust = 0.113, 0.0084 #prior on data gal amplitude

    vec_data_BB  = get_and_select_data_vec(spec_dir, f"cross", spec_name_list, spectra, type, indices)
    samples = mcmc(mean_dust, std_dust, Rminus1_stop=0.01, Rminus1_cl_stop=0.01)
    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(samples, ["a_BB_cmb", "a_BB_dust"], filled=True, title_limit=1)
    plt.savefig(f"{BB_dir}/posterior_BB.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
    
    
    chi2 = -2 * loglike(samples.mean("a_BB_cmb"), samples.mean("a_BB_dust"))
    ndof = n_spec * n_bins - 1
    pte = 1 - ss.chi2(ndof).cdf(chi2)
    print(f"chi2: {chi2}, ndof: {ndof}, pte: {pte}")
    
    lb_ml_BB, vec_ml_BB, cov_ml_BB = get_ML_solution(vec_data_BB, vec_fg_BB, i_cov_BB, n_spec, n_bins, meta_bin_scheme)
    
    
    add_BK = True
    add_sptpol = True
    
    plt.figure(figsize=(12,8))
    plt.plot(l_th, ps_dict["BB"], color="gray")
    plt.ylim(-0.1, 0.25)
    plt.xlim(30, 4000)
    plt.errorbar(lb_ml_BB, vec_ml_BB, np.sqrt(np.diagonal(cov_ml_BB)), fmt="o", color="red", label="ACT DR6")
    if add_BK:
        plt.semilogx()
        lb_bicep, Db_bicep, err_bicep = external_data.get_bicep_BB_spectrum()
        plt.errorbar(lb_bicep, Db_bicep, err_bicep, fmt="o", color="blue", label="BICEP/Keck (2021)")
    if add_sptpol:
        lb_sptpol, Db_sptpol, err_sptpol = external_data.get_sptpol_BB_spectrum()
        plt.errorbar(lb_sptpol, Db_sptpol, err_sptpol, fmt="o", color="orange", label="SPTpol (2020)")


    plt.plot(l_th, ps_dict["BB"] * 0, linestyle="--", color="black")
    plt.legend(fontsize=16)
    plt.ylabel(r"$D_{\ell}^{BB}$", fontsize=22)
    plt.xlabel(r"$\ell$", fontsize=22)
    plt.savefig(f"{BB_dir}/combined_BB.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
    
    
    plt.figure(figsize=(12,8))
    plt.plot(l_th, ps_dict["EE"] * 1/100, label="1% EE", linestyle="--", alpha=0.5)
    plt.plot(l_th, ps_dict["BB"], label="BB", color="gray")
    plt.ylim(-0.1, 0.5)
    plt.xlim(0, 4000)
    plt.errorbar(lb_ml_BB, vec_ml_BB, np.sqrt(np.diagonal(cov_ml_BB)), fmt="o", label="BB ACT DR6", color="red")
    plt.plot(l_th, ps_dict["BB"] * 0, linestyle="--", color="black")
    plt.legend(fontsize=16)
    plt.ylabel(r"$D_{\ell}$", fontsize=22)
    plt.xlabel(r"$\ell$", fontsize=22)
    plt.savefig(f"{BB_dir}/combined_BB_with_EE.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
    
    # also plot individual spectra for the data
    for my_spec in bin_out_dict.keys():
        s_name, spectrum = my_spec
        id, lb = bin_out_dict[my_spec]
        lb_, Db = so_spectra.read_ps(f"{spec_dir}/Dl_{s_name}_cross.dat", spectra=spectra)
        
        plt.figure(figsize=(12,8))
        plt.title(f"{my_spec}, min={np.min(lb)}, max={np.max(lb)}")
        plt.plot(lb_, Db[spectrum], label="original spectrum")
        plt.errorbar(lb, vec_data_BB[id], np.sqrt(cov_BB[np.ix_(id,id)].diagonal()), fmt=".", label="selected spectrum")
        plt.plot(lb, vec_fg_BB[id] + vec_cmb_BB[id], "--", color="gray",alpha=0.3, label="theory")
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig(f"{BB_dir}/{spectrum}_{s_name}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
        
        res = vec_data_BB[id] - vec_fg_BB[id] - vec_cmb_BB[id]
        chi2 = res @ np.linalg.inv(cov_BB[np.ix_(id, id)]) @ res
        ndof = len(lb) - 1
        pte = 1 - ss.chi2(ndof).cdf(chi2)
        print(my_spec, f"chi2: {chi2}, ndof: {ndof}, pte: {pte}")
