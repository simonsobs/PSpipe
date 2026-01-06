import matplotlib

matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_cov
from pspipe_utils import consistency, log
import numpy as np
import pylab as plt
import pickle
import sys
import yaml


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
        consistency.plot_residual(
            lb,
            res_spectrum,
            {"analytical": res_cov},
            "TT",
            f"{map_set} {test}",
            f"{plot_output_dir}/residual_{name}_before",
            lrange=id,
            l_pow=1,
            ylims=calib_infos["y_lims_plot_TT"],
        )

        # Calibrate the spectra
        cal_mean, cal_std = consistency.get_calibration_amplitudes(
            spec_vec,
            full_cov,
            proj_pattern,
            "TT",
            id,
            f"{chains_dir}/{name}",
        )

        results_dict[test, map_set] = {
            "multipole_range": calib_infos["ell_ranges"][map_set],
            "ref_map_set": ref_map_set,
            "calibs": [cal_mean, cal_std],
        }

        calib_vec = np.array([cal_mean**2, cal_mean, 1])
        res_spectrum, res_cov = consistency.project_spectra_vec_and_cov(
            spec_vec, full_cov, proj_pattern, calib_vec=calib_vec
        )

        np.savetxt(
            f"{residual_output_dir}/residual_{name}_after.dat",
            np.array([lb, res_spectrum]).T,
        )
        consistency.plot_residual(
            lb,
            res_spectrum,
            {"analytical": res_cov},
            "TT",
            f"{map_set} {test}",
            f"{plot_output_dir}/residual_{name}_after",
            lrange=id,
            l_pow=1,
            ylims=calib_infos[f"y_lims_plot_TT"],
        )

        fig, ax = plt.subplots(
            2, gridspec_kw={"hspace": 0, "height_ratios": (2, 1)}, figsize=(8, 6)
        )

        proj_indices = get_proj_indices(test)

        ax[0].errorbar(
            lb,
            ps_dict[
                spectra_for_cal[proj_indices[0]][0],
                spectra_for_cal[proj_indices[0]][1],
                "TT",
            ],
            color="grey",
            label=test[:3],
            ls="--",
        )
        ax[0].errorbar(
            lb,
            ps_dict[
                spectra_for_cal[proj_indices[1]][0],
                spectra_for_cal[proj_indices[1]][1],
                "TT",
            ],
            color="blue",
            label=test[-3:],
            ls="--",
        )

        ax[0].errorbar(
            lb,
            ps_dict[
                spectra_for_cal[proj_indices[0]][0],
                spectra_for_cal[proj_indices[0]][1],
                "TT",
            ],
            color="grey",
            label=test[:3],
            ls="--",
        )
        ax[0].errorbar(
            lb,
            ps_dict[
                spectra_for_cal[proj_indices[1]][0],
                spectra_for_cal[proj_indices[1]][1],
                "TT",
            ],
            color="blue",
            label=test[-3:],
            ls="--",
        )

        plt.savefig(f"{plot_output_dir}/calib_full_{name}.png")


# plot the cal factors
color_list = ["blue", "red", "green"]

for i, (map_set, ref_map_set) in enumerate(calib_infos[f"ref_map_sets"].items()):
    print(f"**************")
    print(f"calibration {map_set} with {ref_map_set}")

    for j, test in enumerate(tests):
        cal, std = results_dict[test, map_set]["calibs"]
        print(f"{test}, cal: {cal}, sigma cal: {std}")

        plt.errorbar(
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
        plt.legend(fontsize=15)

x = np.arange(1, len(calib_infos[f"ref_map_sets"]) + 1)
plt.xticks(x, calib_infos[f"ref_map_sets"].keys())
# plt.ylim(0.967, 1.06)
plt.tight_layout()
plt.savefig(f"{plot_output_dir}/calibs_summary.pdf", bbox_inches="tight")
plt.clf()
plt.close()

pickle.dump(results_dict, open(f"{calib_dir}/calibs_dict.pkl", "wb"))
