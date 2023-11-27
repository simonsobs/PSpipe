# This script uses 143/353 GHz spectra from Planck to fit dust amplitude within ACT survey

import argparse
import os

import getdist.plots as gdplt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cobaya.run import run
from getdist.mcsamples import loadMCSamples
from pspipe import conventions
from pspipe_utils import best_fits, consistency
from pspipe_utils import external_data as ext
from pspy import pspy_utils, so_cov, so_dict, so_spectra

matplotlib.use("Agg")

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--use-220", action="store_true", default=False)
parser.add_argument("--dr6-result-path", type=str, default=".")
parser.add_argument("--use-passbands", action="store_true", default=False)
parser.add_argument("--no-fit", action="store_true", default=False)
parser.add_argument("-m", "--mode", type=str, required=True)
args, dict_file = parser.parse_known_args()

use_220 = args.use_220
use_passbands = args.use_passbands
mode = args.mode

d = so_dict.so_dict()
d.read_from_file(dict_file[0])

binning_file = d["binning_file"]
bin_low, bin_high, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax=10_000)


# spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spectra = conventions.spectra
modes = ["TT", "TE", "ET", "EE"] if d["cov_T_E_only"] else spectra

spec_dir = "spectra"
cov_dir = "covariances"

ar = "Planck_f143"
dust_ar = "Planck_f353"

# Build ps and cov dict
array_list = [ar, dust_ar]

# Compute power spectrum of the map residual 353 - 143
ps_template = spec_dir + "/Dl_{}x{}_cross.dat"
cov_template = cov_dir + "/analytic_cov_{}x{}_{}x{}.npy"
ps_dict, cov_dict = consistency.get_ps_and_cov_dict(
    array_list, ps_template, cov_template, spectra_order=spectra
)
lb, ps, cov, _, _ = consistency.compare_spectra(
    array_list, "aa+bb-2ab", ps_dict, cov_dict, mode=mode
)

# Multipole range
lmin, lmax = 300, 2000
idx = np.where((bin_low >= lmin) & (bin_high <= lmax))[0]
res_dict = {"ps": ps[idx], "cov": cov[np.ix_(idx, idx)], "lrange": idx}

# High-ell 220 GHz spectra from ACT DR6
ar220 = "dr6_pa4_f220"
if use_220:
    spec_dir = os.path.join(args.dr6_result_path, "spectra")
    cov_dir = os.path.join(args.dr6_result_path, "covariances")

    lb, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{ar220}x{ar220}_cross.dat", spectra=spectra)

    cov = np.load(f"{cov_dir}/analytic_cov_{ar220}x{ar220}_{ar220}x{ar220}.npy")
    cov = so_cov.selectblock(cov, modes, n_bins=len(lb), block=mode + mode)

    lmin, lmax = 3500, 7125 + 10
    idx = np.where((bin_low >= lmin) & (bin_high <= lmax))[0]

    dict_220 = {"ps": ps[mode][idx], "cov": cov[np.ix_(idx, idx)], "lrange": idx}

if use_passbands:
    npipe_wafers = [ar.replace("Planck", "npipe") for ar in array_list]
    npipe_freq_range = [(50, 1100) for array in array_list]
    npipe_passbands = ext.get_passband_dict_npipe(npipe_wafers, freq_range_list=npipe_freq_range)
    npipe_passbands = {k.replace("npipe", "Planck"): v for k, v in npipe_passbands.items()}

    dr6_wafer = "pa4_f220"
    dr6_passbands = ext.get_passband_dict_dr6([dr6_wafer])
    dr6_passbands = {ar220: dr6_passbands[dr6_wafer]}
else:
    npipe_passbands = {ar: [[d[f"freq_info_{ar}"]["freq_tag"]], [1.0]] for ar in array_list}
    dr6_passbands = {ar220: [[220], [1.0]]}

passbands = npipe_passbands
if use_220:
    passbands.update(**dr6_passbands)


chain_name = f"chains/dust_from_planck353_{mode}/dust"
plot_dir = f"plots/dust_from_planck353_{mode}"
if use_220:
    chain_name = f"{chain_name}_with_220"
    plot_dir = f"{plot_dir}_with_220"
if use_passbands:
    chain_name = f"{chain_name}_passbands"
    plot_dir = f"{plot_dir}_passbands"
pspy_utils.create_directory(plot_dir)

params = {"TT": ["a_c", "a_p", "a_gtt"]}
for m in ["TE", "EE", "BB", "TB"]:
    params[m] = [f"a_g{m.lower()}"]


def compute_fg_ps(ell, fg_dict, exp1, exp2=None):
    if exp2 is None:
        ps = fg_dict[mode.lower(), "all", exp1, exp1]
    else:
        ps = (
            fg_dict[mode.lower(), "all", exp1, exp1]
            + fg_dict[mode.lower(), "all", exp2, exp2]
            - 2 * fg_dict[mode.lower(), "all", exp1, exp2]
        )

    return pspy_utils.naive_binning(ell, ps, binning_file, lmax)


def fit_dust(ell, fg_components, fg_params):
    def loglike(a_p, a_c, a_gtt, a_gte, a_gee, a_gbb, a_gtb):
        fg_params["a_p"] = a_p
        fg_params["a_c"] = a_c
        fg_params["a_gtt"] = a_gtt
        fg_params["a_gte"] = a_gte
        fg_params["a_gee"] = a_gee
        fg_params["a_gbb"] = a_gbb
        fg_params["a_gtb"] = a_gtb

        fg_dict = best_fits.get_foreground_dict(ell, passbands, fg_components, fg_params)

        lrange = res_dict["lrange"]
        lb, ps_res_th = compute_fg_ps(ell, fg_dict, *array_list)

        chi2 = (
            (res_dict["ps"] - ps_res_th[lrange])
            @ np.linalg.inv(res_dict["cov"])
            @ (res_dict["ps"] - ps_res_th[lrange])
        )

        if use_220:
            lrange = dict_220["lrange"]
            lb, ps_220_th = compute_fg_ps(ell, fg_dict, ar220)

            chi2 += (
                (dict_220["ps"] - ps_220_th[lrange])
                @ np.linalg.inv(dict_220["cov"])
                @ (dict_220["ps"] - ps_220_th[lrange])
            )

        return -0.5 * chi2

    info = {
        "likelihood": {"my_like": loglike},
        "sampler": {
            "mcmc": {
                "max_tries": 10_000,
                # "Rminus1_stop": 0.001,
                "Rminus1_stop": 0.05,
                # "Rminus1_cl_stop": 0.05,
            }
        },
        "output": chain_name,
        "force": True,
        "resume": False,
        "debug": False,
        "stop_at_error": True,
    }
    info["params"] = {par: fg_params[par] for par in sum(params.values(), [])}

    priors = {
        "TT": {
            "a_p": {"prior": {"min": 0, "max": 15}, "proposal": 0.1, "latex": "a_p"},
            "a_c": {"prior": {"min": 0, "max": 8}, "proposal": 0.12, "latex": "a_c"},
            "a_gtt": {
                "prior": {"min": 1.0, "max": 20},
                "proposal": 0.1,
                "latex": r"a_\mathrm{dust}^\mathrm{TT}",
            },
        }
    }
    for m in ["TE", "EE", "BB", "TB"]:
        priors[m] = {
            f"a_g{m.lower()}": {
                "prior": {"min": 0, "max": 1},
                "proposal": 0.05,
                "latex": r"a\mathrm{dust}^\mathrm{%s}" % m,
            }
        }
    for key in priors[mode]:
        info["params"][key] = priors[mode][key]

    return run(info)


ell = np.arange(2, lmax + 1)
fg_components = d["fg_components"]
fg_params = d["fg_params"]

if not args.no_fit:
    updated_info, sampler = fit_dust(ell, fg_components, fg_params)

    # Load samples
    # samples = loadMCSamples(chain_name, settings={"ignore_rows": 0.5})
    samples = sampler.products(to_getdist=True, skip_samples=0.5)["sample"]
    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(samples, params[mode], filled=True)
    plt.savefig(f"{plot_dir}/posterior_{mode}.png", dpi=300)

# Plot residuals
bf_fg_params = fg_params.copy()
if not args.no_fit:
    for par_name in params[mode]:
        bf_fg_params[par_name] = samples.mean(par_name)
        print(par_name, bf_fg_params[par_name])

fg_dict = best_fits.get_foreground_dict(ell, passbands, fg_components, bf_fg_params)
lb, fg_th = compute_fg_ps(ell, fg_dict, *array_list)

plt.figure(figsize=(8, 6))
grid = plt.GridSpec(4, 1, hspace=0, wspace=0)

upper = plt.subplot(grid[:3], xticklabels=[], ylabel=r"$D_\ell^{%s}, 353-143}$" % mode)
idx = res_dict["lrange"]
lb, fg_th = lb[idx], fg_th[idx]
ps_res, cov_res = res_dict["ps"], res_dict["cov"]
chi2_res = (ps_res - fg_th) @ np.linalg.inv(cov_res) @ (ps_res - fg_th)

upper.plot(lb, fg_th, color="k")
upper.errorbar(
    lb,
    ps_res,
    np.sqrt(cov_res.diagonal()),
    fmt=".",
    label=r"$\chi^2 = %.2f/%d$" % (chi2_res, len(ps_res)),
)
upper.legend()
if mode == "TT":
    upper.set_yscale("log")

lower = plt.subplot(grid[-1], xlabel=r"$\ell$", ylabel=r"$\Delta D_\ell^{%s}, 353-143}$" % mode)
lower.axhline(0, color="k", ls="--")
lower.errorbar(lb, ps_res - fg_th, np.sqrt(cov_res.diagonal()), ls="None", marker=".")
plt.tight_layout()
plt.savefig(f"{plot_dir}/res_fg_{mode}.png", dpi=300)


if use_220:
    lb, fg_th = compute_fg_ps(ell, fg_dict, ar220)
    idx = dict_220["lrange"]
    lb, fg_th = lb[idx], fg_th[idx]
    ps_220, cov_220 = dict_220["ps"], dict_220["cov"]

    plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(4, 1, hspace=0, wspace=0)

    upper = plt.subplot(grid[:3], xticklabels=[], ylabel=r"$D_\ell^{%s}, 220}$" % mode)
    chi2_220 = (ps_220 - fg_th) @ np.linalg.inv(cov_220) @ (ps_220 - fg_th)

    upper.plot(lb, fg_th, color="k")
    upper.errorbar(
        lb,
        ps_220,
        np.sqrt(cov_220.diagonal()),
        fmt=".",
        label=r"$\chi^2 = %.2f/%d$" % (chi2_220, len(ps_220)),
    )
    upper.legend()

    lower = plt.subplot(grid[-1], xlabel=r"$\ell$", ylabel=r"$\Delta D_\ell^{%s}, 220}$" % mode)
    lower.axhline(0, color="k", ls="--")
    lower.errorbar(lb, ps_220 - fg_th, np.sqrt(cov_220.diagonal()), ls="None", marker=".")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/res_220_{mode}.png", dpi=300)
