"""
This script computes differences between split
power spectra for the different arrays specified
in the paramfile. 
"""
from pspipe_utils import consistency
from pspy import so_dict, pspy_utils
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy.stats as ss
import pickle as p

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

ps_dir = "spectra"
cov_dir = "split_covariances"

output_dir = "plots/split_nulls"
pspy_utils.create_directory(output_dir)

covariances_type = [
    "analytic_cov",
    #"analytic_cov_with_mc_corrections",
    "analytic_cov_with_diag_mc_corrections"
]

surveys = d["surveys"]

arrays = [f"{sv}_{ar}" for sv in surveys for ar in d[f"arrays_{sv}"]]
n_splits = {ar: len(d[f"maps_{ar}"]) for ar in arrays}

modes = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


multipole_range = {
    "dr6_pa4_f150": {
        "T": [1250, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa4_f220": {
        "T": [1000, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa5_f090": {
        "T": [800, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa5_f150": {
        "T": [800, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa6_f090": {
        "T": [600, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa6_f150": {
        "T": [600, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    }
}

chi2_dict = {cov_type: {} for cov_type in covariances_type}
pte_dict = {cov_type: {} for cov_type in covariances_type}

for ar in arrays:

    for cov_type in covariances_type:
        chi2_dict[cov_type][ar] = {m: [] for m in modes}
        pte_dict[cov_type][ar] = {m: [] for m in modes}

    ns = n_splits[ar]
    splits_id = [str(i) for i in range(ns)]

    cross_splits = combinations(splits_id, 2)
    split_diff_list = combinations(cross_splits, 2)

    ps_template = f"{ps_dir}/Dl_{ar}x{ar}_" + "{}{}.dat"
    name = f"{ar}" + "_{}"
    cov_dict = {}
    for cov_type in covariances_type:
        cov_template = f"{cov_dir}/{cov_type}_{name}x{name}_{name}x{name}.npy"
        ps_dict, cov = consistency.get_ps_and_cov_dict(splits_id, ps_template, cov_template, spectra_order = modes, skip_auto=True)
        cov_dict[cov_type] = cov

    for (s1, s2), (s3, s4) in split_diff_list:

        split_list = [s1, s2, s3, s4]

        for m in modes:

            m0, m1 = m[0], m[1]
            lmin0, lmax0 = multipole_range[ar][m0]
            lmin1, lmax1 = multipole_range[ar][m1]
            lmin = max(lmin0, lmin1)
            lmax = min(lmax0, lmax1)

            res_cov_dict = {}
            for cov_type in covariances_type:
                lb, res_ps, res_cov, _, _ = consistency.compare_spectra(split_list, "ab-cd", ps_dict, cov_dict[cov_type], mode=m)
                res_cov_dict[cov_type] = res_cov

            lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
            for cov_type in covariances_type:
                cov = res_cov_dict[cov_type][np.ix_(lrange, lrange)]
                chi2 = res_ps[lrange] @ np.linalg.inv(cov) @ res_ps[lrange]
                pte = 1 - ss.chi2(len(lb[lrange])).cdf(chi2)

                chi2_dict[cov_type][ar][m].append(chi2)
                chi2_dict[cov_type][ar, m, "ndof"] = len(lb[lrange])

                pte_dict[cov_type][ar][m].append(pte)

            plt.figure(figsize=(8, 6))
            plt.xlabel(r"$\ell$")
            plt.ylabel(r"$\Delta D_\ell^{%s}$" % m)
            if m == "TT":
                plt.ylabel(r"$\ell\Delta D_\ell^{TT}$")
            elif m == "EE":
                plt.ylabel(r"${\ell}^{-1}\Delta D_\ell^{EE}$")
            plt.axhline(0, color="k", ls="--")

            ell_scaling = 0
            if m == "TT":
                ell_scaling = 1
            elif m == "EE":
                ell_scaling = -1
            rescale_amp = lb ** ell_scaling

            plt.plot(lb, rescale_amp*res_ps, alpha=0.)
            ylims = plt.gca().get_ylim()

            for cov_type in covariances_type[::-1]:
                res_cov = res_cov_dict[cov_type]
                chi2 = chi2_dict[cov_type][ar][m][-1]

                plt.errorbar(lb, rescale_amp*res_ps, rescale_amp*np.sqrt(res_cov.diagonal()), marker="o",
                            label=f"[{cov_type}] $\chi^2/d.o.f. = {chi2:.1f}/{len(lb[lrange])}$",
                            #color="tab:blue", markeredgecolor="tab:blue",
                            markerfacecolor="white",
                            lw=0.7, elinewidth=1.3)
            plt.axvspan(0, lb[lrange][0], color="gray", alpha=0.4)
            plt.ylim(*ylims)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/null_{s1}{s2}-{s3}{s4}_{ar}_{m}.png")

p.dump(chi2_dict, open(f"{output_dir}/chi2_dict.pkl", "wb"))
p.dump(pte_dict, open(f"{output_dir}/pte_dict.pkl", "wb"))

for cov_type in covariances_type:
    for ar in arrays:
        for m in modes:
            plt.figure(figsize=(8, 6))

            ndof = chi2_dict[cov_type][ar, m, "ndof"]
            plt.hist(np.array(chi2_dict[cov_type][ar][m]), bins=15, density=True)
            plt.axvline(ndof, color="k", ls="--")

            x = np.linspace(ndof/4, 2*ndof, 500)
            y = ss.chi2(ndof).pdf(x)
            plt.plot(x, y)

            plt.title(f"{ar} split null - {m}")
            plt.xlabel(r"$\chi^2$")

            plt.tight_layout()
            plt.savefig(f"{output_dir}/{ar}_{m}_{cov_type}_split_null.png", dpi=300)

# PTE summary plot
x_list = np.arange(len(arrays))

cmap = plt.get_cmap("tab10")
spacing = np.linspace(-0.23, 0.23, len(modes))
modes = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

delta_x = {m: spacing[i] for i,m in enumerate(modes)}
colors = {m: cmap(i) for i,m in enumerate(modes)}

for cov_type in covariances_type:
    plt.figure(figsize=(11, 5))
    for i, ar in enumerate(arrays):
        for m in modes:

            y = pte_dict[cov_type][ar][m]
            x = np.full_like(y, x_list[i] + delta_x[m])

            plt.scatter(x, y, color=colors[m], alpha=0.4, label=m, s=45)
        if i == 0:
            plt.legend(fontsize=13, ncol=3, loc="upper center")

    plt.ylabel(r"Probability to exceed (PTE)")
    plt.xticks(x_list, arrays)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pte_summary_{cov_type}.png", dpi=300)

# Histogram
for cov_type in covariances_type:
    pte_array = np.array([pte_dict[cov_type][ar][m] for ar in arrays for m in modes])
    pte_array = pte_array.flatten()

    n_samples = len(pte_array)
    n_bins = 20
    bins = np.linspace(0,1,n_bins+1)

    plt.figure(figsize=(8,6))
    plt.xlabel(r"Probability to exceed (PTE)")
    plt.hist(pte_array, bins=bins)
    plt.axhline(n_samples/n_bins, color="k", ls="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pte_hist_{cov_type}.png", dpi=300)
