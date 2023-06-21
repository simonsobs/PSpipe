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

surveys = d["surveys"]

arrays = [f"{sv}_{ar}" for sv in surveys for ar in d[f"arrays_{sv}"]]
n_splits = {ar: len(d[f"maps_{ar}"]) for ar in arrays}

modes = ["TT", "TE", "ET", "EE"]

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

chi2_dict = {}
pte_dict = {}
for ar in arrays:

    chi2_dict[ar] = {m: [] for m in modes}
    pte_dict[ar] = {m: [] for m in modes}
    ns = n_splits[ar]
    splits_id = [str(i) for i in range(ns)]

    cross_splits = combinations(splits_id, 2)
    split_diff_list = combinations(cross_splits, 2)
    print(list(split_diff_list))
    ps_template = f"{ps_dir}/Dl_{ar}x{ar}_" + "{}{}.dat"
    name = f"{ar}" + "_{}"
    cov_template = f"{cov_dir}/analytic_cov_{ar}x{ar}_{ar}x{ar}" + "_{}{}x{}{}.npy"

    ps_dict, cov_dict = consistency.get_ps_and_cov_dict(splits_id, ps_template, cov_template, skip_auto=True)

    for (s1, s2), (s3, s4) in split_diff_list:

        split_list = [s1, s2, s3, s4]

        for m in modes:

            m0, m1 = m[0], m[1]
            lmin0, lmax0 = multipole_range[ar][m0]
            lmin1, lmax1 = multipole_range[ar][m1]
            lmin = max(lmin0, lmin1)
            lmax = min(lmax0, lmax1)

            lb, res_ps, res_cov, chi2, pte = consistency.compare_spectra(split_list, "ab-cd", ps_dict, cov_dict, mode=m)

            lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
            chi2 = res_ps[lrange] @ np.linalg.inv(res_cov[np.ix_(lrange, lrange)]) @ res_ps[lrange]

            chi2_dict[ar][m].append(chi2)
            chi2_dict[ar, m, "ndof"] = len(lb[lrange])

            pte = 1 - ss.chi2(len(lb[lrange])).cdf(chi2)
            pte_dict[ar][m].append(pte)

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
            plt.errorbar(lb, rescale_amp*res_ps, rescale_amp*np.sqrt(res_cov.diagonal()), marker="o",
                         label=f"$\chi^2/d.o.f. = {chi2:.1f}/{len(lb[lrange])}$",
                         color="tab:blue", markeredgecolor="tab:blue", markerfacecolor="white",
                         lw=0.7, elinewidth=1.3)
            plt.axvspan(0, lb[lrange][0], color="gray", alpha=0.4)
            plt.ylim(*ylims)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/null_{s1}{s2}-{s3}{s4}_{ar}_{m}.png")

p.dump(chi2_dict, open(f"{output_dir}/chi2_dict.pkl", "wb"))
p.dump(pte_dict, open(f"{output_dir}/pte_dict.pkl", "wb"))

for ar in arrays:
    for m in modes:
        plt.figure(figsize=(8, 6))

        ndof = chi2_dict[ar, m, "ndof"]
        plt.hist(np.array(chi2_dict[ar][m]), bins=15, density=True)
        plt.axvline(ndof, color="k", ls="--")

        x = np.linspace(ndof/4, 2*ndof, 500)
        y = ss.chi2(ndof).pdf(x)
        plt.plot(x, y)

        plt.title(f"{ar} split null - {m}")
        plt.xlabel(r"$\chi^2$")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ar}_{m}_split_null.png", dpi=300)

# PTE summary plot
x_list = np.arange(len(arrays))
colors = {"TT": "tab:blue", "TE": "tab:orange", "ET": "tab:green", "EE": "tab:red"}
delta_x = {"TT": -0.195, "TE": -0.065, "ET": 0.065, "EE": 0.195}

plt.figure(figsize=(8, 6))
for i, ar in enumerate(arrays):
    for m in ["TT", "TE", "ET", "EE"]:

        y = pte_dict[ar][m]
        x = np.full_like(y, x_list[i] + delta_x[m])

        plt.scatter(x, y, color=colors[m], alpha=0.4, label=m,s=65)
    if i == 0:
        plt.legend(fontsize=13, ncol=4, loc="upper center")

plt.ylabel(r"Probability to exceed (PTE)")
plt.xticks(x_list, arrays)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{output_dir}/pte_summary.png", dpi=300)
