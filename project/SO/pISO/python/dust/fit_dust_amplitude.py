"""
This script uses 143/353 GHz spectra from Planck to fit dust amplitude within the deep56 patch
if --use-220 is used , we will also fit the high ell 220 GHz ACT channel, the idea being to try to break the degeneracy between dust and CIB. In the default case, we use beta_c = beta_p = 2.20, which is preferred by Planck data. To set a different value use the flag --beta_value, while if you want to sample the beta CIB use the flag --sample_beta
example use:
python fit_dust_amplitude.py global_dust.dict --mode BB
python fit_dust_amplitude.py global_dust.dict --mode TT --use-220  --dr6-result-path ./dr6
"""


import argparse
import os
import getdist.plots as gdplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from cobaya.run import run
from getdist import loadMCSamples
from pspipe_utils import best_fits
from pspy import pspy_utils, so_dict
import dust_utils
mpl.use("Agg")

def run_mcmc(chain_name, fg_params, Rminus1_stop, Rminus1_cl_stop):
    
    info = {
        "likelihood": {"my_like": loglike},
        "sampler": {
            "mcmc": {
                "max_tries": 10_000,
                "Rminus1_stop": Rminus1_stop,
                "Rminus1_cl_stop": Rminus1_cl_stop,
            }
        },
        "output": chain_name,
        "force": True,
        "resume": False,
        "debug": False,
        "stop_at_error": True,
    }
    
    info["params"] = {par: fg_params[par] for par in sum(params.values(), []) if par in fg_params}

    priors = {
        "TT": {
            "a_p": {"prior": {"min": 0, "max": 15}, "proposal": 0.1, "latex": "a_p"},
            "a_c": {"prior": {"min": 0, "max": 15}, "proposal": 0.12, "latex": "a_c"},
            "a_gtt": { "prior": {"min": 0.0, "max": 20}, "proposal": 0.1, "latex": r"a_\mathrm{dust}^\mathrm{TT}"},
        }
    }
    
    if args.sample_beta:
        # when sampling beta for CIB, forcing beta_c and beta_p to be equal (no difference seen when
        # they are sampled differently)
        priors["TT"].update(
            {
             "beta_p": {"prior": {"min": 0, "max": 5}, "proposal": 0.1, "latex": r"\beta_p"},
             "beta_c": {"value": 'lambda beta_p: beta_p', "derived": True, "latex": r"\beta_c"},
             })

     
    if use_220:
        priors["TT"].update(
            {
                f"bandint_shift_dr6_pa4_f220": {
                    "prior": {"min": -50, "max": 50},
                    "proposal": 4,
                    "latex": r"\Delta_{\rm band}^{220}",
                }
            }
        )

    for m in ["TE", "EE", "BB", "TB"]:
        priors[m] = {
            f"a_g{m.lower()}": {
                "prior": {"min": 0, "max": 1},
                "proposal": 0.05,
                "latex": r"a_\mathrm{dust}^\mathrm{%s}" % m,
            }
        }
        
    for key in priors[mode]:
        info["params"][key] = priors[mode][key]

    return run(info)
    
    
def loglike(a_p, a_c, a_gtt, a_gte, a_gee, a_gbb, a_gtb, beta_c = 2.2, beta_p = 2.2, bandint_shift_dr6_pa4_f220=0.0):
    """
    Compute the loglikelihood for the given fg parameters
    The residual from which we measure the dust is by default the Planck 353 GHz x353 GHz + 143 GHz x143  GHz - 2 x 143 GHz x353 GHz residual
    optionnaly if use_220 has been set to TRUE, we will also fit the high ell 220 GHz ACT channel
    """
    # reassign the beta values if not sampled
    if args.sample_beta:
        beta_c = beta_c
        beta_p = beta_p
    else:
        beta_c = args.beta_value
        beta_p = args.beta_value

    if use_220:
        _, res_planck_model, _, fg_220_model = get_fg_model(a_p, a_c, a_gtt, a_gte, a_gee, a_gbb, a_gtb, beta_c, beta_p, bandint_shift_dr6_pa4_f220=bandint_shift_dr6_pa4_f220)
                                                              
        chi2 =  (res_planck[idx_planck] - res_planck_model[idx_planck]) @ icov_res_planck @ (res_planck[idx_planck] - res_planck_model[idx_planck])
        chi2 +=  (ps_220[idx_220] - fg_220_model[idx_220]) @ icov_220 @ (ps_220[idx_220] - fg_220_model[idx_220])
        
    else:
        _, res_planck_model = get_fg_model(a_p, a_c, a_gtt, a_gte, a_gee, a_gbb, a_gtb, beta_c, beta_p)
        
        chi2 =  (res_planck[idx_planck] - res_planck_model[idx_planck]) @ icov_res_planck @ (res_planck[idx_planck] - res_planck_model[idx_planck])

    return -0.5 * chi2
        
def get_fg_model(a_p, a_c, a_gtt, a_gte, a_gee, a_gbb, a_gtb, beta_c, beta_p, bandint_shift_dr6_pa4_f220=0.0):
    """
    get the fg model for the given fg parameters
    The residual from which we measure the dust is by default the Planck 353 GHz x353 GHz + 143 GHz x143  GHz - 2 x 143 GHz x353 GHz residual
    optionnaly if use_220 has been set to TRUE, we will also fit the high ell 220 GHz ACT channel
    """
    
    fg_params["a_p"] = a_p
    fg_params["a_c"] = a_c
    fg_params["a_gtt"] = a_gtt
    fg_params["a_gte"] = a_gte
    fg_params["a_gee"] = a_gee
    fg_params["a_gbb"] = a_gbb
    fg_params["a_gtb"] = a_gtb
    fg_params["beta_p"] = beta_p
    fg_params["beta_c"] = beta_c

    band_shift_dict = {}
    for map_set in passbands.keys():
        band_shift_dict[f"bandint_shift_{map_set}"] =  0.0
            
    band_shift_dict[f"bandint_shift_dr6_pa4_f220"] = bandint_shift_dr6_pa4_f220
        
    fg_dict = best_fits.get_foreground_dict(
        ell, passbands, fg_components, fg_params, band_shift_dict=band_shift_dict
    )
    
    fg_res = (
            fg_dict[mode.lower(), "all", "Planck_f143", "Planck_f143"]
            + fg_dict[mode.lower(), "all",  "Planck_f353", "Planck_f353"]
            - 2 * fg_dict[mode.lower(), "all", "Planck_f143", "Planck_f353"]
    )
    
    lb, fg_res = pspy_utils.naive_binning(ell, fg_res, binning_file, lmax)
    
    if use_220:
        lb_220, fg_220 = pspy_utils.naive_binning(ell,
                                                  fg_dict[mode.lower(), "all", "dr6_pa4_f220", "dr6_pa4_f220"],
                                                  binning_file,
                                                  lmax)
        return lb, fg_res, lb_220, fg_220
    else:
        return lb, fg_res
        

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--use-220", action="store_true", default=False) 
parser.add_argument("--dr6-result-path-spectra", type=str, default=".") 
parser.add_argument("--dr6-result-path-covariance", type=str, default=".")
parser.add_argument("--no-fit", action="store_true", default=False) 
parser.add_argument("-m", "--mode", type=str, required=True)
parser.add_argument("--leak-corr", action="store_true", default=False)
parser.add_argument("--sample_beta", action="store_true", default=False)
parser.add_argument("--beta_value", type=float, default=2.2)
args, dict_file = parser.parse_known_args()

mode = args.mode
use_220 = args.use_220 and mode == "TT"

d = so_dict.so_dict()
d.read_from_file(dict_file[0])

binning_file = d["binning_file"]
do_bandpass_integration = d["do_bandpass_integration"]
Rminus1_stop = 0.05
Rminus1_cl_stop = 0.1
mc_cov = True

result_dir = d["chain_dir"] + f"/dust_from_planck353_{mode}" 
plot_dir = d["plots_dir"] + f"/dust_from_planck353_{mode}" 
if args.leak_corr:
    result_dir += "_leak_corr"
    plot_dir += "_leak_corr"

chain_name = f"{result_dir}/dust"

if not args.leak_corr:
    spec_dir = d["spec_dir"]
    leak_cov = False
else:
    spec_dir = d["spectra_leak_corr_planck_bias_corr_dir"]
    leak_cov = True

if use_220:
    chain_name += "_with_220"
    plot_dir += "_with_220"
    
if do_bandpass_integration:
    chain_name += f"_passbands"
    plot_dir += f"_passbands"
    
if args.sample_beta:
    chain_name += f"_sampled_beta"
    plot_dir += f"_sampled_beta"
else:
    chain_name += f"_beta{args.beta_value}"
    plot_dir += f"_beta{args.beta_value}"

pspy_utils.create_directory(plot_dir)

passbands = dust_utils.load_band_pass(d, use_220=use_220)

bin_low, bin_high, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax=10_000)

if d["cov_T_E_only"] == False:
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
else:
    spectra = ["TT", "TE", "ET", "EE"]

lmin, lmax = 300, 2000
idx_planck = np.where((bin_low >= lmin) & (bin_high <= lmax))[0]
lb, res_planck, cov_res_planck = dust_utils.get_residual_and_cov(["Planck_f143", "Planck_f353"],
                                                                 spec_dir, 
                                                                 d["cov_dir"], 
                                                                 mode,
                                                                 spectra,
                                                                 op="aa+bb-2ab",
                                                                 mc_cov=mc_cov,
                                                                 leak_cov=leak_cov)

icov_res_planck = np.linalg.inv(cov_res_planck[np.ix_(idx_planck, idx_planck)]) #pre-invert the matrix to avoid doing it during MCMC

# High-ell 220 GHz spectra from ACT DR6
if use_220:
    spec_name = "dr6_pa4_f220xdr6_pa4_f220"
    spec_dir = os.path.join(args.dr6_result_path_spectra)
    cov_dir = os.path.join(args.dr6_result_path_covariance)
    lb, ps_220, cov_220 = dust_utils.get_spectra_and_cov(spec_dir, cov_dir, spec_name, mode, spectra, mc_cov=mc_cov, leak_cov=leak_cov)
    lmin, lmax = 4500, 8500+10
    idx_220 = np.where((bin_low >= lmin) & (bin_high <= lmax))[0]
    icov_220 = np.linalg.inv(cov_220[np.ix_(idx_220, idx_220)])

params = {"TT": ["a_c", "a_p", "a_gtt"]}
if use_220:
    params["TT"].append(f"bandint_shift_dr6_pa4_f220")
    
if args.sample_beta:
    params["TT"].append("beta_c")
    params["TT"].append("beta_p")

for m in ["TE", "EE", "BB", "TB"]:
    params[m] = [f"a_g{m.lower()}"]

ell = np.arange(2, lmax + 1)
fg_components = d["fg_components"]
fg_params = d["fg_params"]
print(fg_params)


if args.no_fit:
    # Load samples
    samples = loadMCSamples(chain_name, settings={"ignore_rows": 0.5})
else:
    updated_info, sampler = run_mcmc(chain_name, fg_params, Rminus1_stop, Rminus1_cl_stop)
    samples = sampler.products(to_getdist=True, skip_samples=0.5)["sample"]

g = gdplt.get_subplot_plotter()
g.triangle_plot(samples, params[mode], filled=True, title_limit=1)
# Add a nice table if needed but latex required (@NERSC do `module load texlive`)
# with mpl.rc_context(rc={"text.usetex": True}):
#     table = samples.getTable(limit=1, paramList=params[mode])
#     g.subplots[0, 0].text(1.2, 0, table.tableTex().replace("\n", ""), size=15, ha="right")
plt.savefig(f"{plot_dir}/posterior_{mode}.png", dpi=300)

# Plot residuals
mean = fg_params.copy()
for par_name in params[mode]:
    mean[par_name] = samples.mean(par_name)

if use_220:
    lb, res_planck_model, lb_220, fg_220_model = get_fg_model(mean["a_p"], mean["a_c"], mean["a_gtt"], mean["a_gte"], mean["a_gee"], mean["a_gbb"], mean["a_gtb"], mean["beta_c"], mean["beta_p"], bandint_shift_dr6_pa4_f220=mean["bandint_shift_dr6_pa4_f220"])

                                                  
    chi2 =  (res_planck[idx_planck] - res_planck_model[idx_planck]) @ icov_res_planck @ (res_planck[idx_planck] - res_planck_model[idx_planck])
    chi2 +=  (ps_220[idx_220] - fg_220_model[idx_220]) @ icov_220 @ (ps_220[idx_220] - fg_220_model[idx_220])
    
    ndof = len(idx_planck) + len(idx_220) - len(params[mode])
    pte = 1 - ss.chi2(ndof).cdf(chi2)

else:
    lb, res_planck_model = get_fg_model(mean["a_p"], mean["a_c"], mean["a_gtt"], mean["a_gte"],
                                        mean["a_gee"], mean["a_gbb"], mean["a_gtb"], mean["beta_c"], mean["beta_p"], bandint_shift_dr6_pa4_f220=0)


    chi2 =  (res_planck[idx_planck] - res_planck_model[idx_planck]) @ icov_res_planck @ (res_planck[idx_planck] - res_planck_model[idx_planck])
    
    ndof = len(idx_planck) - len(params[mode])
    pte = 1 - ss.chi2(ndof).cdf(chi2)

print(f"mode : {mode}, params: {params[mode]}, chi2: {chi2:.02f}, ndof:{ndof}, pte:{pte:.02f}")

y_lim = {}
y_lim["TT"] = [4500, 7000]
y_lim["EE"] = [-300, 300]
y_lim["TE"] = [-100, 400]
y_lim["BB"] = [-300, 300]
y_lim["TB"] = [-300, 300]


err_res_planck =  np.sqrt(cov_res_planck.diagonal())
np.savetxt(f"{result_dir}/residual.dat", np.transpose([lb[idx_planck], res_planck[idx_planck], err_res_planck[idx_planck], res_planck_model[idx_planck]]))
np.save(f"{result_dir}/icov_residual.npy", icov_res_planck)


fig = plt.figure(figsize=(8, 6))
grid = plt.GridSpec(4, 1, hspace=0, wspace=0)
upper = fig.add_subplot(grid[:3], xticklabels=[], ylabel=r"$D_\ell^{%s, 353-143}$" % mode)
upper.plot(lb[idx_planck], res_planck_model[idx_planck], color="k")
upper.errorbar(lb[idx_planck],
               res_planck[idx_planck],
               err_res_planck[idx_planck],
               fmt=".",
               label=r"$\chi^2 = %.2f/%d$ - PTE = %.3f" % (chi2, ndof, pte))
               
               
upper.set_ylim(y_lim[mode])
upper.legend()
if mode == "TT":
    upper.set_yscale("log")

lower = fig.add_subplot(grid[-1], xlabel=r"$\ell$", ylabel=r"$\Delta D_\ell^{%s, 353-143}$" % mode)
lower.axhline(0, color="k", ls="--")
lower.errorbar(lb[idx_planck], res_planck[idx_planck] - res_planck_model[idx_planck], np.sqrt(cov_res_planck[np.ix_(idx_planck, idx_planck)].diagonal()), ls="None", marker=".")
fig.tight_layout()
fig.savefig(f"{plot_dir}/res_fg_{mode}.png", dpi=300)
if use_220:
    fig = plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(4, 1, hspace=0, wspace=0)
    upper = fig.add_subplot(grid[:3], xticklabels=[], ylabel=r"$D_\ell^{%s, 220}$" % mode)
    upper.plot(lb[idx_220], fg_220_model[idx_220], color="k")
    upper.errorbar(lb[idx_220], ps_220[idx_220], np.sqrt(cov_220[np.ix_(idx_220, idx_220)].diagonal()), fmt=".")

    lower = fig.add_subplot(grid[-1], xlabel=r"$\ell$", ylabel=r"$\Delta D_\ell^{%s, 220}$" % mode)
    lower.axhline(0, color="k", ls="--")
    lower.errorbar(lb[idx_220], ps_220[idx_220] - fg_220_model[idx_220], np.sqrt(cov_220[np.ix_(idx_220, idx_220)].diagonal()), ls="None", marker=".")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/res_220_{mode}.png", dpi=300)
