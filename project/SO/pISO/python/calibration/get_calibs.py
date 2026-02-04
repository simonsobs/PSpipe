import matplotlib

matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_cov
from pspipe_utils import consistency, log
import numpy as np
import pylab as plt
import pickle
import sys
import yaml
import scipy.stats as ss


def get_proj_pattern(test, map_set, ref_map_set):
    if test == "AxA-AxB":
        proj_pattern = np.array([1, -1, 0])
        #    [1, -1, 0]]  x [ AxA, AxB, BxB].T    =  AxA - AxB
        name = f"{map_set}x{map_set}-{map_set}x{ref_map_set}"
    elif test == "AxA-BxB":
        proj_pattern = np.array([1, 0, -1])
        #    [1, 0, -1]]  x [ AxA, AxB, BxB].T    =   AxA - BxB
        name = f"{map_set}x{map_set}-{ref_map_set}x{ref_map_set}"
    elif test == "BxB-AxB":
        proj_pattern = np.array([0, -1, 1])
        #    [[0, -1, 1]]   x [ AxA, AxB, BxB].T    =    BxB - AxB
        name = f"{ref_map_set}x{ref_map_set}-{map_set}x{ref_map_set}"
    return name, proj_pattern


def get_proj_indices(test):
    if test == "AxA-AxB":
        return (0, 1)
    elif test == "AxA-BxB":
        return (0, 2)
    elif test == "BxB-AxB":
        return (2, 1)


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# log calib infos from calib yaml file
with open(d["calib_yaml"], "r") as f:
    calib_dict: dict = yaml.safe_load(f)
calib_infos: dict = calib_dict["get_calibs.py"]

planck_corr = calib_infos["planck_corr"]
subtract_bf_fg = calib_infos["subtract_bf_fg"]

data_dir = d["data_dir"]
plot_dir = d["plots_dir"]
spec_dir = d["spec_dir"]
cov_dir = d["cov_dir"]
bestfit_dir = d["best_fits_dir"]

# Create output dirs
calib_dir = d["calib_dir"]
residual_output_dir = f"{calib_dir}/residuals"
chains_dir = f"{calib_dir}/chains"
plot_output_dir = plot_dir + "/calib/"

if planck_corr:
    spec_dir = "spectra_leak_corr_planck_bias_corr"
    calib_dir += "_planck_bias_corrected"

if subtract_bf_fg:
    calib_dir += "_fg_sub"

_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
n_bins = len(lb)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes = ["TT", "TE", "ET", "EE"]
else:
    modes = spectra

pspy_utils.create_directory(calib_dir)
pspy_utils.create_directory(residual_output_dir)
pspy_utils.create_directory(chains_dir)
pspy_utils.create_directory(plot_output_dir)

# Define the projection pattern - i.e. which
# spectra combination will be used to compute
# the residuals with
#   A: the array you want to calibrate
#   B: the reference array

results_dict = {}

tests = ["AxA-AxB", "AxA-BxB", "BxB-AxB"]


for test in tests:
    pte_list = []
    pte_cal_list = []
    # Loop over each map_set-ref_map_set combination
    # (so you can measure calib on a same map_set with different ref_map_set)
    for map_set, ref_map_set in calib_infos[f"ref_map_sets"].items():
        name, proj_pattern = get_proj_pattern(test, map_set, ref_map_set)

        spectra_for_cal = [
            (ref_map_set, ref_map_set, "TT"),
            (ref_map_set, map_set, "TT"),
            (map_set, map_set, "TT"),
            # (map_set, map_set, "TT"),
            # (map_set, ref_map_set, "TT"),
            # (ref_map_set, ref_map_set, "TT"),
        ]

        # Load spectra and cov
        ps_dict = {}
        cov_dict = {}
        for i, (ms1, ms2, m1) in enumerate(spectra_for_cal):
            _, ps = so_spectra.read_ps(
                f"{spec_dir}/Dl_{ms1}x{ms2}_cross.dat", spectra=spectra
            )

            ps_dict[ms1, ms2, m1] = ps[m1]

            if (m1 == "TT") & (subtract_bf_fg):
                log.info(f"remove fg {m1}  {ms1} x {ms2}")
                l_fg, bf_fg = so_spectra.read_ps(
                    f"{bestfit_dir}/fg_{ms1}x{ms2}.dat", spectra=spectra
                )
                _, bf_fg_TT_binned = pspy_utils.naive_binning(
                    l_fg, bf_fg["TT"], d["binning_file"], d["lmax"]
                )
                ps_dict[ms1, ms2, m1] -= bf_fg_TT_binned

            for j, (ms3, ms4, m2) in enumerate(spectra_for_cal):
                if j < i:
                    continue
                cov = np.load(f"{cov_dir}/analytic_cov_{ms1}x{ms2}_{ms3}x{ms4}.npy")
                cov = so_cov.selectblock(cov, modes, n_bins=n_bins, block=m1 + m2)
                cov_dict[(ms1, ms2, m1), (ms3, ms4, m2)] = cov

        # Concatenation
        spec_vec, full_cov = consistency.append_spectra_and_cov(
            ps_dict, cov_dict, spectra_for_cal
        )

        # Save and plot residuals before calibration
        res_spectrum, res_cov = consistency.project_spectra_vec_and_cov(
            spec_vec, full_cov, proj_pattern
        )
        np.savetxt(
            f"{residual_output_dir}/residual_{name}_before.dat",
            np.array([lb, res_spectrum]).T,
        )
        np.savetxt(f"{residual_output_dir}/residual_cov_{name}.dat", res_cov)

        lmin, lmax = calib_infos["ell_ranges"][map_set]
        id = np.where((lb >= lmin) & (lb <= lmax))[0]

        # Calibrate the spectra
        cal_mean, cal_std = consistency.get_calibration_amplitudes(
            spec_vec,
            full_cov,
            proj_pattern,
            "TT",
            id,
            f"{chains_dir}/{name}",
            Rminus1_cl_stop=0.01,
        )

        results_dict[test, map_set] = {
            "multipole_range": calib_infos["ell_ranges"][map_set],
            "ref_map_set": ref_map_set,
            "calibs": [cal_mean, cal_std],
        }

        calib_vec = np.array([cal_mean**2, cal_mean, 1])
        res_spectrum_cal, res_cov_cal = consistency.project_spectra_vec_and_cov(
            spec_vec, full_cov, proj_pattern, calib_vec=calib_vec
        )

        np.savetxt(
            f"{residual_output_dir}/residual_{name}_after.dat",
            np.array([lb, res_spectrum_cal]).T,
        )
        
        ### THIS REPLACES CONSISTENCY.PLOT_RESIDUAL
        expected_res = 0.
        remove_dof = 0.
        
        res_th = np.ones(len(lb)) * expected_res
        chi2 = (res_spectrum[id] - res_th[id]) @ np.linalg.inv(res_cov[np.ix_(id, id)]) @ (res_spectrum[id] - res_th[id])
        ndof = len(lb[id]) - remove_dof
        pte = 1 - ss.chi2(ndof).cdf(chi2)
        pte_list.append(pte)
        
        chi2_cal = (res_spectrum_cal[id] - res_th[id]) @ np.linalg.inv(res_cov_cal[np.ix_(id, id)]) @ (res_spectrum_cal[id] - res_th[id])
        ndof_cal = len(lb[id]) - remove_dof - 2  # TODO : Should there be a - 2 here ??
        pte_cal = 1 - ss.chi2(ndof_cal).cdf(chi2_cal)
        pte_cal_list.append(pte_cal)
        fig, ax = plt.subplots(
            2,
            gridspec_kw={"hspace": 0, "height_ratios": (2.5, 1)},
            figsize=(9, 9),
            sharex=True,
            dpi=200,
        )

        ax[0].set_yscale("log")

        proj_indices = get_proj_indices(test)
        ell_pow = 1
        ax[0].errorbar(
            lb - 6,
            ps_dict[
                spectra_for_cal[proj_indices[0]][0],
                spectra_for_cal[proj_indices[0]][1],
                "TT",
            ] * lb**ell_pow,
            np.sqrt(
                cov_dict[
                    (
                        spectra_for_cal[proj_indices[0]][0],
                        spectra_for_cal[proj_indices[0]][1],
                        "TT",
                    ),
                    (
                        spectra_for_cal[proj_indices[0]][0],
                        spectra_for_cal[proj_indices[0]][1],
                        "TT",
                    ),
                ].diagonal()
            ) * lb**ell_pow,
            color="tab:red",
            label=test[-3:],
            ls="",
            alpha=.6,
            marker='.',
            mfc='white',
            mec='tab:red',
            linewidth=.5,
            elinewidth=.6,
        )
        ax[0].errorbar(
            lb + 2,
            ps_dict[
                spectra_for_cal[proj_indices[1]][0],
                spectra_for_cal[proj_indices[1]][1],
                "TT",
            ] * lb**ell_pow,
            np.sqrt(
                cov_dict[
                    (
                        spectra_for_cal[proj_indices[1]][0],
                        spectra_for_cal[proj_indices[1]][1],
                        "TT",
                    ),
                    (
                        spectra_for_cal[proj_indices[1]][0],
                        spectra_for_cal[proj_indices[1]][1],
                        "TT",
                    ),
                ].diagonal()
            ) * lb**ell_pow,
            color="tab:blue",
            label=test[:3],
            ls="",
            alpha=.6,
            marker='.',
            mfc='white',
            mec='tab:blue',
            linewidth=.5,
            elinewidth=.6,
        )

        ax[0].errorbar(
            lb - 2,
            ps_dict[
                spectra_for_cal[proj_indices[0]][0],
                spectra_for_cal[proj_indices[0]][1],
                "TT",
            ]
            / calib_vec[proj_indices[1]]
            * lb**ell_pow,
            np.sqrt(
                cov_dict[
                    (
                        spectra_for_cal[proj_indices[0]][0],
                        spectra_for_cal[proj_indices[0]][1],
                        "TT",
                    ),
                    (
                        spectra_for_cal[proj_indices[0]][0],
                        spectra_for_cal[proj_indices[0]][1],
                        "TT",
                    ),
                ].diagonal()
            ) * calib_vec[proj_indices[0]]
            * lb**ell_pow,
            color="red",
            label=test[-3:] + ' cal',
            ls="-",
            marker='.',
            mfc='white',
            mec='red',
            linewidth=.5,
            elinewidth=.6,
            alpha=.8,
        )
        ax[0].errorbar(
            lb + 6,
            ps_dict[
                spectra_for_cal[proj_indices[1]][0],
                spectra_for_cal[proj_indices[1]][1],
                "TT",
            ]
            / calib_vec[proj_indices[0]]
            * lb**ell_pow,
            np.sqrt(
                cov_dict[
                    (
                        spectra_for_cal[proj_indices[1]][0],
                        spectra_for_cal[proj_indices[1]][1],
                        "TT",
                    ),
                    (
                        spectra_for_cal[proj_indices[1]][0],
                        spectra_for_cal[proj_indices[1]][1],
                        "TT",
                    ),
                ].diagonal()
            ) * calib_vec[proj_indices[1]]
            * lb**ell_pow,
            color="blue",
            label=test[:3]+ ' cal',
            ls="-",
            marker='.',
            mfc='white',
            mec='blue',
            alpha=.8,
            linewidth=.7,
            elinewidth=.6,
        )
        
        ax[1].errorbar(lb, - res_spectrum / np.sqrt(res_cov.diagonal()),
                     yerr=np.sqrt(res_cov.diagonal()) / np.sqrt(res_cov.diagonal()),
                     ls="None", marker = ".",
                     linewidth=.5,
                     alpha=.5,
                     color="tab:blue",
                    #  label=f"{test} [$\chi^2 = {{{chi2:.1f}}}/{{{ndof}}}$ (${{{pte:.4f}}}$)]")
                     label=f"{test} [PTE={pte:.4f}]")
        ax[1].errorbar(lb, - res_spectrum_cal / np.sqrt(res_cov_cal.diagonal()),
                     yerr=np.sqrt(res_cov_cal.diagonal()) / np.sqrt(res_cov_cal.diagonal()),
                     ls="None", marker = ".",
                     color="blue",
                    #  label=f"{test} cal [$\chi^2={{{chi2_cal:.1f}}}/{{{ndof_cal}}}$ (${{{pte_cal:.4f}}}$)]")
                     label=f"{test} cal [PTE={pte_cal:.4f}]")
        ax[1].axhline(0, color='black', zorder=-10)

        ax[1].axvspan(xmin=0, xmax=lmin,
                    color="gray", alpha=0.7, zorder=-20)
        ax[1].axvspan(xmin=lmax, xmax=10000,
                    color="gray", alpha=0.7, zorder=-20)

        ax[0].legend(title=f'A={map_set}   B={ref_map_set}')
        ax[0].set_xlim(0, lmax + 600)
        ax[0].set_ylim(*calib_infos['ylims'])
        ax[0].set_ylabel(fr"$\ell^{{{ell_pow}}} D_\ell^\mathrm{{TT}}$", fontsize=18)
        ax[0].set_title(f'cal={cal_mean:.3f}+-{cal_std:.3f}')

        ax[1].legend(loc='lower right')
        ax[1].set_ylim(-8, 6)
        ax[1].set_ylabel(fr"$\Delta D_\ell^\mathrm{{TT}} / \sigma(\Delta D_\ell^\mathrm{{TT}})$", fontsize=16)
        ax[1].set_xlabel(r"$\ell$", fontsize=20)

        plt.savefig(f"{plot_output_dir}/calib_full_{name}.png")
        plt.close()
        
    n_samples = len(pte_list)
    n_bins_hist = 10
    bins = np.linspace(0, 1, n_bins_hist + 1)
    min_pte, max_pte = np.min(pte_list), np.max(pte_list)
    min_pte_cal, max_pte_cal = np.min(pte_cal_list), np.max(pte_cal_list)
    
    plt.figure(figsize=(8,6))
    plt.title("Array-bands test", fontsize=16)
    plt.xlabel(r"Probability to exceed (PTE)", fontsize=16)
    plt.hist(pte_list, bins=bins, label=f"n tests: {n_samples}, min: {min_pte:.3f}, max: {max_pte:.3f}", histtype="bar", facecolor="blue", edgecolor="black", linewidth=3, alpha=.7)
    plt.hist(pte_cal_list, bins=bins, label=f"n tests: {n_samples}, min: {min_pte_cal:.3f}, max: {max_pte_cal:.3f}", histtype="bar", facecolor="red", edgecolor="black", linewidth=3, alpha=.7)
    plt.axhline(n_samples/n_bins_hist, color="k", ls="--", alpha=0.5)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.savefig(f"{plot_output_dir}/PTE_hist_{test}", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


# plot the cal factors
color_list = ["blue", "red", "green"]
fig, ax = plt.subplots(figsize=(12, 6))
for i, (map_set, ref_map_set) in enumerate(calib_infos[f"ref_map_sets"].items()):
    print(f"**************")
    print(f"calibration {map_set} with {ref_map_set}")
    ax.axhline(1, color='grey', ls='--')
    for j, test in enumerate(tests):
        cal, std = results_dict[test, map_set]["calibs"]
        print(f"{test}, cal: {cal:.5f}, sigma cal: {std:.5f}")

        ax.errorbar(
            i + 0.9 + j * 0.1,
            cal,
            std,
            label=test,
            color=color_list[j],
            marker=".",
            ls="None",
            markersize=6.5,
            markeredgewidth=2,
        )

    if i == 0:
        ax.legend(fontsize=15)

x = np.arange(1, len(calib_infos[f"ref_map_sets"]) + 1)
ax.set_xticks(x, calib_infos[f"ref_map_sets"].keys())
ax.set_ylabel('Calibration factor', fontsize=18)
# plt.ylim(0.967, 1.06)
plt.tight_layout()
plt.savefig(f"{plot_output_dir}/calibs_summary.pdf", bbox_inches="tight")
plt.clf()
plt.close()

print(results_dict[test, map_set]["calibs"][0])
calibs_to_save = {
    'bestfits' : {
        sv_ar: {
            test: float(results_dict[test, sv_ar]["calibs"][0])
            for test in tests
        }
        for sv_ar in calib_infos[f"ref_map_sets"].keys()
    },
    'std' : {
        sv_ar: {
            test: float(results_dict[test, sv_ar]["calibs"][1])
            for test in tests
        }
        for sv_ar in calib_infos[f"ref_map_sets"].keys()
    },
}

file = open(f"{calib_dir}/calibs_dict.yaml", "w")
yaml.dump(calibs_to_save, file)
file.close()