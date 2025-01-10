"""
This script compute split null tests on
simulations and build the expected PTE histogram
"""
from pspipe_utils import consistency
from pspy import so_dict, pspy_utils
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import scipy.stats as ss

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

sim_spec_dir = d["sim_spec_dir"]
cov_dir = "split_covariances"

output_dir = "split_nulls"
pspy_utils.create_directory(output_dir)

covariances_type = [
    "analytic_cov",
    "analytic_cov_with_mc_corrections",
    "analytic_cov_with_diag_mc_corrections"
]

surveys = d["surveys"]
iStart = d["iStart"]
iStop = d["iStop"]

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

pte_list = {cov_type: [] for cov_type in covariances_type}

for iii in range(iStart, iStop+1):
    print(f"Split null for sim {iii}")
    temp_pte_list = {cov_type: [] for cov_type in covariances_type}

    for ar in arrays:

        ns = n_splits[ar]
        splits_id = [str(i) for i in range(ns)]

        cross_splits = combinations(splits_id, 2)
        split_diff_list = combinations(cross_splits, 2)

        ps_template = f"{sim_spec_dir}/Dl_{ar}x{ar}_" + "{}{}" + f"_{iii:05d}.dat"
        name = f"{ar}" + "_{}"
        cov_dict = {}
        for cov_type in covariances_type:
            cov_template = f"{cov_dir}/{cov_type}_{name}x{name}_{name}x{name}.npy"
            ps_dict, cov = consistency.get_ps_and_cov_dict(splits_id, ps_template, cov_template, skip_auto=True)
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
                    temp_pte_list[cov_type].append(pte)

    for cov_type in covariances_type:
        pte_list[cov_type].append(temp_pte_list[cov_type])

pickle.dump(pte_list, open(f"{output_dir}/mc_pte_dict.pkl", "wb"))

plt.figure(figsize=(8,6))
plt.xlabel(r"Probability to exceed (PTE)")

for cov_type in covariances_type:

    pte_array = pte_list[cov_type]

    n_samples = len(pte_array[0])
    n_bins = 20

    bins = np.linspace(0, 1, n_bins + 1)
    bin_center = (bins[1:] + bins[:-1]) / 2

    counts = [np.histogram(pte_array[i], bins=bins)[0] for i in range(iStart, iStop + 1)]
    counts_mean = np.mean(counts, axis=0)
    counts_std = np.std(counts, axis=0)

    plt.errorbar(bin_center, counts_mean, counts_std, ls="None",
                 marker="o", label=cov_type)
    plt.axhline(n_samples/n_bins, color="k", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/mc_pte_hist.png")
