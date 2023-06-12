from pspipe_utils import consistency
from pspy import so_dict, pspy_utils
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy.stats as ss

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

ps_dir = "spectra"
cov_dir = "split_covariances"

output_dir = "plots/split_nulls"
pspy_utils.create_directory(output_dir)

surveys = d["surveys"]

arrays = [f"{sv}_{ar}" for sv in surveys for ar in d[f"arrays_{sv}"]]
n_splits = {ar: len(d[f"maps_{ar}"]) for ar in arrays}

modes = ["TT", "TE", "ET", "EE"]

multipole_range = {90: {"T": [800, 8500],
                        "E": [500, 8500]},
                   150: {"T": [1250, 8500],
                         "E": [500, 8500]},
                   220: {"T": [1000, 8500],
                         "E": [500, 8500]}}

multipole_range = {
    "dr6_pa4_f150": {
        "T": [1250, 8500],
        "E": [500, 8500]
    },
    "dr6_pa4_f220": {
        "T": [1000, 8500],
        "E": [500, 8500]
    },
    "dr6_pa5_f090": {
        "T": [800, 8500],
        "E": [500, 8500]
    },
    "dr6_pa5_f150": {
        "T": [800, 8500],
        "E": [500, 8500]
    },
    "dr6_pa6_f090": {
        "T": [600, 8500],
        "E": [500, 8500]
    },
    "dr6_pa6_f150": {
        "T": [600, 8500],
        "E": [500, 8500]
    }
}

#multipole_range = {90: {"T": [1200, 7000],
#                        "E": [300, 7000]},
#                   150: {"T": [1250, 7000],
#                         "E": [300, 7000]},
#                   220: {"T": [1500, 7000],
#                         "E": [300, 7000]}}

chi2_dict = {}
pte_dict = {}
for ar in arrays:

    chi2_dict[ar] = {m: [] for m in modes}
    pte_dict[ar] = {m: [] for m in modes}
    ns = n_splits[ar]
    splits_id = [str(i) for i in range(ns)]

    cross_splits = list(combinations(splits_id, 2))
    split_diff_list = list(combinations(cross_splits, 2))
    print(split_diff_list)
    ps_template = f"{ps_dir}/Dl_{ar}x{ar}_" + "{}{}.dat"
    name = f"{ar}" + "_{}"
    cov_template = f"{cov_dir}/analytic_cov_{ar}x{ar}_{ar}x{ar}" + "_{}{}x{}{}.npy"
    #cov_template = f"{cov_dir}/analytic_cov_with_mc_corrections_{name}x{name}_{name}x{name}.npy"
    #cov_template = f"{cov_dir}/analytic_cov_with_diag_mc_corrections_{name}x{name}_{name}x{name}.npy"
    ps_dict, cov_dict = consistency.get_ps_and_cov_dict(splits_id, ps_template, cov_template, skip_auto = True)

    # test feature
    res_dict_to_plot = {m: [] for m in modes}
    invcov_dict_to_plot = {m: [] for m in modes}
    # end of test
    for (s1, s2), (s3, s4) in split_diff_list:

        split_list = [s1, s2, s3, s4]

        for m in modes:

            m0, m1 = m[0], m[1]
            #lmin0, lmax0 = multipole_range[f][m0]
            #lmin1, lmax1 = multipole_range[f][m1]
            lmin0, lmax0 = multipole_range[ar][m0]
            lmin1, lmax1 = multipole_range[ar][m1]
            lmin = max(lmin0, lmin1)
            lmax = min(lmax0, lmax1)

            lb, res_ps, res_cov, chi2, pte = consistency.compare_spectra(split_list, "ab-cd", ps_dict, cov_dict, mode = m)

            lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
            chi2 = res_ps[lrange] @ np.linalg.inv(res_cov[np.ix_(lrange, lrange)]) @ res_ps[lrange]

            # test feature
            res_dict_to_plot[m].append((lb[lrange], res_ps[lrange]))
            invcov_dict_to_plot[m].append(np.linalg.inv(res_cov[np.ix_(lrange, lrange)]))
            # end of test

            chi2_dict[ar][m].append(chi2)
            chi2_dict[ar, m, "ndof"] = len(lb[lrange])

            pte = 1 - ss.chi2(len(lb[lrange])).cdf(chi2)
            pte_dict[ar][m].append(pte)

            plt.figure(figsize = (8, 6))
            plt.xlabel(r"$\ell$")
            plt.ylabel(r"$\Delta D_\ell^{%s}$" % m)
            if m == "TT":
                plt.ylabel(r"$\ell\Delta D_\ell^{TT}$")
            elif m == "EE":
                plt.ylabel(r"${\ell}^{-1}\Delta D_\ell^{EE}$")
            plt.axhline(0, color = "k", ls = "--")

            ell_scaling = 0
            if m == "TT":
                ell_scaling = 1
            elif m == "EE":
                ell_scaling = -1
            rescale_amp = lb ** ell_scaling

            # ylims ?
            plt.plot(lb, rescale_amp*res_ps, alpha=0.)
            ylims = plt.gca().get_ylim()
            plt.errorbar(lb, rescale_amp*res_ps, rescale_amp*np.sqrt(res_cov.diagonal()), marker = "o",
                         label = f"$\chi^2/d.o.f. = {chi2:.1f}/{len(lb[lrange])}$",
                         color = "tab:blue", markeredgecolor = "tab:blue", markerfacecolor = "white",
                         lw = 0.7, elinewidth = 1.3)
            plt.axvspan(0, lb[lrange][0], color = "gray", alpha = 0.4)
            plt.ylim(*ylims)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/null_{s1}{s2}-{s3}{s4}_{ar}_{m}.png")

    # test feature
    for m in modes:
        plt.figure(figsize = (8, 6))
        mean_res = []
        mean_cov = np.linalg.inv(np.sum(invcov_dict_to_plot[m], axis = 0))
        plt.axhline(0, color = "k", ls = "--")
        for i, (lb, res) in enumerate(res_dict_to_plot[m]):
            plt.plot(lb, res, color = "gray", alpha = 0.4, lw = 0.8)
            mean_res.append(res)
        mean_res = mean_cov @ np.sum([invcov_dict_to_plot[m][i] @ res_dict_to_plot[m][i][1] for i in range(len(mean_res))], axis = 0)
        plt.errorbar(lb, mean_res, yerr = np.sqrt(mean_cov.diagonal()), color = "tab:red", lw = 1.2,
                     marker = "o", markerfacecolor = "white", markeredgecolor = "tab:red")
        ymin, ymax = (mean_res - 7.5*np.sqrt(mean_cov.diagonal())).min(), (mean_res + 7.5*np.sqrt(mean_cov.diagonal())).max()
        plt.ylim(ymin, ymax)
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$\Delta D_\ell^{%s}$" % m)
        plt.title(f"{ar} split null - {m}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ar}_{m}_residuals.png", dpi = 300)
    # end of test

import pickle as p
p.dump(chi2_dict, open(f"{output_dir}/chi2_dict.pkl", "wb"))
p.dump(pte_dict, open(f"{output_dir}/pte_dict.pkl", "wb"))

# test
import scipy.stats as ss
for ar in arrays:
    for m in modes:
        plt.figure(figsize = (8, 6))

        ndof = chi2_dict[ar, m, "ndof"]
        plt.hist(np.array(chi2_dict[ar][m]), bins = 15, density = True)
        plt.axvline(ndof, color = "k", ls = "--")

        # test
        x = np.linspace(ndof/4, 2*ndof, 500)
        y = ss.chi2(ndof).pdf(x)
        plt.plot(x, y)

        plt.title(f"{ar} split null - {m}")
        plt.xlabel(r"$\chi^2$")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ar}_{m}_split_null.png", dpi = 300)
