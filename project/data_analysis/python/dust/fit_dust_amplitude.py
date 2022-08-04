from itertools import combinations_with_replacement as cwr
from pspy import so_dict, pspy_utils
from pspy import so_spectra, so_cov
import matplotlib.pyplot as plt
from pspipe_utils import consistency, best_fits
import numpy as np
import sys
import os
from cobaya.run import run
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

spec_dir = "spectra"
cov_dir = "covariances"

ar = "Planck_f143"
dust_ar = "Planck_f353"

mode = "TE"

use220 = False
spec_dir_220 = "spectra"
cov_dir_220 = "covariances"

ar220 = "dr6_pa4_f220"

# Multipole range
lmin_res, lmax_res = 300, 2000
lmin_220, lmax_220 = 3500, 7000

chain_name = f"chains_dust_from_planck353_{mode}/dust"
output_dir = f"plots/dust_from_planck353_{mode}"
pspy_utils.create_directory(output_dir)


# Build ps and cov dict
array_list = [ar, dust_ar]
ps_list = []
for i, ar1 in enumerate(array_list):
    for j, ar2 in enumerate(array_list):
        if j < i: continue
        ps_list.append((ar1, ar2, mode))

ps_dict = {}
cov_dict = {}
for i, (ar1, ar2, m1) in enumerate(ps_list):
    lb_res, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{ar1}x{ar2}_cross.dat", spectra = spectra)
    ps_dict[ar1, ar2, m1] = ps[m1]
    for j, (ar3, ar4, m2) in enumerate(ps_list):
        if j < i: continue
        cov = np.load(f"{cov_dir}/analytic_cov_{ar1}x{ar2}_{ar3}x{ar4}.npy")
        cov = so_cov.selectblock(cov, modes, n_bins = len(lb_res),
                                 block = m1 + m2)
        cov_dict[(ar1, ar2, m1), (ar3, ar4, m2)] = cov


# Compute power spectrum of the map residual 353 - 143
ps_vec, full_cov = consistency.append_spectra_and_cov(ps_dict, cov_dict,
                                                         [(ar, ar, mode),
                                                          (ar, dust_ar, mode),
                                                          (dust_ar, dust_ar, mode)])

ps_res, cov_res = consistency.project_spectra_vec_and_cov(ps_vec, full_cov, [1, -2, 1])
id_res = np.where((lb_res >= lmin_res) & (lb_res <= lmax_res))[0]
lb_res, ps_res, cov_res = lb_res[id_res], ps_res[id_res], cov_res[np.ix_(id_res, id_res)]

if use220:
    # Load high-ell 220 GHz spectra from ACT DR6
    lb_220, ps_220 = so_spectra.read_ps(f"{spec_dir_220}/Dl_{ar220}x{ar220}_cross.dat", spectra = spectra)
    ps_220 = ps_220[mode]

    cov_220 = np.load(f"{cov_dir_220}/analytic_cov_{ar220}x{ar220}_{ar220}x{ar220}.npy")
    cov_220 = so_cov.selectblock(cov_220, modes, n_bins = len(lb_220),
                                 block = mode + mode)

    id_220 = np.where((lb_220 >= lmin_220) & (lb_220 <= lmax_220))[0]
    lb_220, ps_220, cov_220 = lb_220[id_220], ps_220[id_220], cov_220[np.ix_(id_220, id_220)]

def model_res(ell, fg_components, fg_params, mode, binning_file):


    fg_dict = best_fits.get_foreground_dict(ell, [143, 353],
                                            fg_components, fg_params)

    fg_143 = fg_dict[mode.lower(), "all", 143, 143]
    fg_353 = fg_dict[mode.lower(), "all", 353, 353]
    fg_cross = fg_dict[mode.lower(), "all", 143, 353]
    fg_res = fg_143 + fg_353 - 2 * fg_cross

    lmax = ell[-1]
    lb, fg_res_b = pspy_utils.naive_binning(ell, fg_res, binning_file, lmax)

    return lb, fg_res_b

def model_220(ell, fg_components, fg_params, mode, binning_file):

    fg_dict = best_fits.get_foreground_dict(ell, [220],
                                            fg_components, fg_params)
    fg = fg_dict[mode.lower(), "all", 220, 220]

    lmax = ell[-1]
    lb, fg_b = pspy_utils.naive_binning(ell, fg, binning_file, lmax)

    return lb, fg_b

def fit_dust(ell, ps_res, cov_res, id_res, mode, binning_file, chain_name,
             fg_components, fg_params, ps_220 = None, cov_220 = None, id_220 = None):

    def loglike(a_p, a_c, a_gtt, a_gte, a_gee):

        fg_params["a_p"] = a_p
        fg_params["a_c"] = a_c
        fg_params["a_gtt"] = a_gtt
        fg_params["a_gte"] = a_gte
        fg_params["a_gee"] = a_gee

        lb_fg, ps_res_th = model_res(ell, fg_components, fg_params, mode, binning_file)
        lb_fg, ps_res_th = lb_fg[id_res], ps_res_th[id_res]

        chi2 = (ps_res - ps_res_th) @ np.linalg.inv(cov_res) @ (ps_res - ps_res_th)

        if ps_220 is not None:
            lb_220, ps_220_th = model_220(ell, fg_components, fg_params, mode, binning_file)
            lb_220, ps_220_th = lb_220[id_220], ps_220_th[id_220]

            chi2 += (ps_220 - ps_220_th) @ np.linalg.inv(cov_220) @ (ps_220 - ps_220_th)

        return -0.5 * chi2

    info = {
        "likelihood": {"my_like": loglike},
        "sampler": {
            "mcmc": {
                "max_tries": 1e4,
                "Rminus1_stop": 0.005,
                "Rminus1_cl_stop": 0.05,
                    }
                   },
        "output": chain_name,
        "force": True,
        "resume": False
           }
    info["params"] = {"a_p": fg_params["a_p"],
                      "a_c": fg_params["a_c"],
                      "a_gtt": fg_params["a_gtt"],
                      "a_gte": fg_params["a_gte"],
                      "a_gee": fg_params["a_gee"]}

    priors = {"TT": {
                  "a_p": {"prior": {"min": 0, "max": 15},
                          "proposal": 0.1,
                          "latex": "a_p"},
                  "a_c":{"prior": {"min": 0, "max": 8},
                         "proposal": 0.12,
                         "latex": "a_c"},
                  "a_gtt": {"prior": {"min": 1.0, "max": 20},
                            "proposal": 0.1,
                            "latex": "a_\mathrm{dust}^\mathrm{TT}"}},
              "TE": {
                  "a_gte": {"prior": {"min": 0, "max": 1},
                                   "proposal": 0.05,
                                   "latex": "a_\mathrm{dust}^\mathrm{TE}"}},
              "EE": {
                  "a_gee": {"prior": {"min": 0, "max": 1},
                                   "proposal": 0.03,
                                   "latex": "a_\mathrm{dust}^\mathrm{EE}"}},
              }
    for key in priors[mode]:
        info["params"][key] = priors[mode][key]

    updated_info, sampler = run(info)
    pars = list(priors[mode].keys())

    samples = loadMCSamples(chain_name, settings = {"ignore_rows": 0.5})
    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(samples, pars, filled = True)
    plt.savefig(f"{output_dir}/posterior_{mode}.png", dpi = 300)

ell = np.arange(2, 7001)
fg_components = {"tt": ["tSZ_and_CIB", "cibp", "kSZ", "radio", "dust"],
                 "te": ["radio", "dust"],
                 "ee": ["radio", "dust"],
                 "bb": ["radio", "dust"],
                 "tb": ["radio", "dust"],
                 "eb": []}

fg_params = {"a_tSZ": 3.30, "a_kSZ": 1.60, "a_p": 5.97, "beta_p": 2.2,
             "a_c": 6.32, "beta_c": 2.20, "a_s": 3.10, "a_gtt": 13.86,
             "a_gte": 0.69, "a_gee": 0.27, "a_psee": 0.05, "a_pste": 0,
             "a_gbb": 0, "a_psbb": 0, "a_gtb": 0, "a_pstb": 0,
             "xi": 0.1, "T_d": 9.60}
if use220:
    fit_dust(ell, ps_res, cov_res, id_res, mode, d["binning_file"],
             chain_name, fg_components, fg_params,
             ps_220, cov_220, id_220)
else:
    fit_dust(ell, ps_res, cov_res, id_res, mode, d["binning_file"],
             chain_name, fg_components, fg_params)

params = {"TT": ["a_c", "a_p", "a_gtt"],
          "TE": ["a_gte"],
          "EE": ["a_gee"]}

samples = loadMCSamples(chain_name, settings = {"ignore_rows": 0.5})
bf_fg_params = fg_params.copy()
for par_name in params[mode]:
    bf_fg_params[par_name] = samples.mean(par_name)

lb_res_th, fg_res_th = model_res(ell, fg_components, fg_params, mode, d["binning_file"])
id_plot_theory = np.where((lb_res_th >= lmin_res) & (lb_res_th <= lmax_res))

plt.figure(figsize = (8, 6))
grid = plt.GridSpec(4, 1, hspace = 0, wspace = 0)


upper = plt.subplot(grid[:3], xticklabels = [], ylabel = r"$D_\ell^{%s}, 353-143}$" % mode)
chi2_res = (ps_res - fg_res_th[id_res]) @ np.linalg.inv(cov_res) @ (ps_res - fg_res_th[id_res])
ndof_res = len(ps_res)
upper.plot(lb_res_th[id_plot_theory], fg_res_th[id_plot_theory], color = "k")
upper.errorbar(lb_res, ps_res, np.sqrt(cov_res.diagonal()), ls = "None", marker = ".",
               label = r"$\chi^2 = %.2f/%d$" % (chi2_res, ndof_res))
upper.legend()
if mode == "TT":
    upper.set_yscale("log")

lower = plt.subplot(grid[-1], xlabel = r"$\ell$", ylabel = r"$\Delta D_\ell^{%s}, 353-143}$" % mode)
lower.axhline(0, color = "k", ls = "--")
lower.errorbar(lb_res, ps_res - fg_res_th[id_res], np.sqrt(cov_res.diagonal()),
               ls = "None", marker = ".")
plt.tight_layout()
plt.savefig(f"{output_dir}/res_fg_{mode}.png", dpi = 300)



if use220:
    lb_220_th, fg_220_th = model_220(ell, fg_components, fg_params, mode, d["binning_file"])
    id_plot_theory = np.where((lb_220_th >= lmin_220) & (lb_220_th <= lmax_220))
    plt.figure(figsize = (8, 6))
    grid = plt.GridSpec(4, 1, hspace = 0, wspace = 0)


    upper = plt.subplot(grid[:3], xticklabels = [], ylabel = r"$D_\ell^{%s}, 220}$" % mode)
    chi2_220 = (ps_220 - fg_220_th[id_220]) @ np.linalg.inv(cov_220) @ (ps_220 - fg_220_th[id_220])
    ndof_220 = len(ps_220)
    upper.plot(lb_220_th[id_plot_theory], fg_220_th[id_plot_theory], color = "k")
    upper.errorbar(lb_220, ps_220, np.sqrt(cov_220.diagonal()), ls = "None", marker = ".",
                   label = r"$\chi^2 = %.2f/%d$" % (chi2_220, ndof_220))
    upper.legend()

    lower = plt.subplot(grid[-1], xlabel = r"$\ell$", ylabel = r"$\Delta D_\ell^{%s}, 220}$" % mode)
    lower.axhline(0, color = "k", ls = "--")
    lower.errorbar(lb_220, ps_220 - fg_220_th[id_220], np.sqrt(cov_220.diagonal()),
                   ls = "None", marker = ".")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/res_220_{mode}.png", dpi = 300)
