"""
This script performs null tests between different arrays
and plot residual power spectra and a summary PTE distribution
"""

from pspy import so_dict, pspy_utils, so_cov
from itertools import combinations_with_replacement as cwr
from pspipe_utils import consistency, best_fits
import pickle
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"

output_dir = "plots/array_nulls"
pspy_utils.create_directory(output_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

surveys = d["surveys"]
arrays = []
for sv in surveys:
    for ar in d[f"arrays_{sv}"]:
        arrays.append(f"{sv}_{ar}")

# Select the spectra and cov to use
# should write a dictionnary like
# {label: (spectra_dir, cov_type)}
nulls_data = {
    "MC beam and leakage corrections": ("spectra_corrected", "analytic_cov_with_diag_mc_beam_and_leakage_corrections"),
    "MC corrections": ("spectra", "analytic_cov_with_diag_mc_corrections"),
}

# Load ps and covariance dict
ps_dict = {}
cov_dict = {}
for label, (spec_dir, cov_type) in nulls_data.items():
    ps_template = spec_dir + "/Dl_{}x{}_cross.dat"
    cov_template = f"{cov_dir}/{cov_type}" + "_{}x{}_{}x{}.npy"
    ps, cov = consistency.get_ps_and_cov_dict(arrays, ps_template, cov_template, spectra_order=spectra)
    ps_dict[label] = ps
    lb = ps["ell"]
    cov_dict[label] = cov


# Load foreground best fits
fg_file_name = "best_fits/fg_{}x{}.dat"
lth, fg_dict = best_fits.fg_dict_from_files(fg_file_name, arrays, d["lmax"], spectra=spectra)

# Define the multipole range
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

# Options
remove_first_bin=True
l_pows = {
    "TT": 1,
    "TE": 0,
    "TB": 0,
    "ET": 0,
    "BT": 0,
    "EE": -1,
    "EB": -1,
    "BE": -1,
    "BB": -1
}
y_lims = {
    "TT": (-100000, 75000),
    "TE": (-30, 30),
    "TB": (-30, 30),
    "ET": (-30, 30),
    "BT": (-30, 30),
    "EE": (-0.01, 0.005),
    "EB": (-0.01, 0.005),
    "BE": (-0.01, 0.005),
    "BB": (-0.01, 0.005)
}
pte_threshold = 0.01

# Define PTE dict
pte_dict = {label: [] for label in nulls_data}

operations = {"diff": "ab-cd"}

for i, (ar1, ar2) in enumerate(cwr(arrays, 2)):
    for j, (ar3, ar4) in enumerate(cwr(arrays, 2)):

        if j <= i: continue
        f1, f2 = d[f"freq_info_{ar1}"]["freq_tag"], d[f"freq_info_{ar2}"]["freq_tag"]
        f3, f4 = d[f"freq_info_{ar3}"]["freq_tag"], d[f"freq_info_{ar4}"]["freq_tag"]

        ar_list = [ar1, ar2, ar3, ar4]

        plot_title = f"{ar1}x{ar2} - {ar3}x{ar4}"
        expected_res = 0.

        for m in spectra:

            # Select the multipoles
            m0, m1 = m[0], m[1]
            if (f1 != f3) and (m0 == "T"): continue
            if (f2 != f4) and (m1 == "T"): continue
            #if f"{ar1}x{ar2}".count("pa4_f220") != f"{ar3}x{ar4}".count("pa4_f220"): continue

            lmin0, lmax0 = multipole_range[ar1][m0]
            lmin1, lmax1 = multipole_range[ar2][m1]
            ps12_lmin = max(lmin0, lmin1)
            ps12_lmax = min(lmax0, lmax1)

            lmin2, lmax2 = multipole_range[ar3][m0]
            lmin3, lmax3 = multipole_range[ar4][m1]
            ps34_lmin = max(lmin2, lmin3)
            ps34_lmax = min(lmax2, lmax3)

            lmin = max(ps12_lmin, ps34_lmin)
            lmax = min(ps12_lmax, ps34_lmax)

            # Compute residuals
            res_ps_dict = {}
            res_cov_dict = {}
            for label in nulls_data:
                lb, res_ps, res_cov, _, _ = consistency.compare_spectra(ar_list, "ab-cd", ps_dict[label], cov_dict[label], mode = m)
                if remove_first_bin:
                    # remove first bin
                    lb, res_ps, res_cov = lb[1:], res_ps[1:], res_cov[1:,1:]
                res_ps_dict[label] = res_ps
                res_cov_dict[label] = res_cov

                # plot res corrmat
                corr = so_cov.cov2corr(res_cov)
                plt.figure(figsize=(9,8))
                ax=plt.gca()
                centers = [lb[0], lb[-1], lb[-1], lb[0]]
                dx, = np.diff(centers[:2]) / (corr.shape[1]-1)
                dy, = np.diff(centers[2:]) / (corr.shape[0]-1)
                extent=[centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
                im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", extent=extent, aspect="auto")
                plt.title(f"{ar1}x{ar2} - {ar3}x{ar4}")
                divider=make_axes_locatable(ax)
                cax=divider.append_axes("right", size="5%",pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.tight_layout()
                fname = f"corr_{label}_{ar1}x{ar2}_{ar3}x{ar4}_{m}.png"
                plt.savefig(f"{output_dir}/{fname}", dpi=300)

            lrange = np.where((lb >= lmin) & (lb <= lmax))[0]

            # Compute theory residual from bestfits
            res_th = fg_dict[ar1, ar2][m] - fg_dict[ar3, ar4][m]
            lb_fg, res_th_b = pspy_utils.naive_binning(lth, res_th, d["binning_file"], d["lmax"])
            # remove first bin
            lb_fg, res_th_b = lb_fg[1:], res_th_b[1:]

            # Plot residual and get chi2
            fname = f"diff_{ar1}x{ar2}_{ar3}x{ar4}_{m}"
            chi2_dict = consistency.plot_residual(
                lb, res_ps_dict, res_cov_dict, mode=m,
                title=plot_title.replace('dr6_', ''),
                file_name=f"{output_dir}/{fname}",
                expected_res=expected_res,
                lrange=lrange,
                overplot_theory_lines=(lb_fg, res_th_b),
                l_pow=l_pows[m],
                return_chi2=True,
                ylims=y_lims[m]
            )

            # Fill pte_dict
            for label in nulls_data:
                pte = 1-ss.chi2(chi2_dict[label]["ndof"]).cdf(chi2_dict[label]["chi2"])
                pte_dict[label].append(pte)
                if (pte <= pte_threshold) or (pte >= 1-pte_threshold):
                    print(f"[{label}] [{plot_title} {m}] PTE = {pte:.03f}")

# Save pte to pickle
pickle.dump(pte_dict, open(f"{output_dir}/pte_dict.pkl", "wb"))

# Plot PTE histogram for each cov type
for label in nulls_data:

    pte_list = pte_dict[label]
    n_samples = len(pte_list)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)

    plt.figure(figsize=(8,6))
    plt.xlabel(r"Probability to exceed (PTE)")

    plt.hist(pte_list, bins=bins)
    plt.axhline(n_samples/n_bins, color="k", ls="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pte_hist_{label}.png", dpi=300)
