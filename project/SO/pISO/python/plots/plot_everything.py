"""
Plots all kinds of spectra of combination A x B
"""

from pspy import so_spectra, pspy_utils, so_cov, so_map, so_window, so_dict
from pspipe_utils import log
from math import pi
import numpy as np
import healpy as hp

# import pylab as plt
from matplotlib import pyplot as plt
import os
from pspy import pspy_utils
from matplotlib.colors import Normalize
import itertools
import yaml
import sys
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spectra_auto = ["TT", "EE", "BB"]


def bin_array(array, binning: int = None):
    """Bin a given array.
        If the array size is not a multiple of the binning,
        ignore the rest of the array.

    Args:
        array (_type_): _description_
        binning (int): _description_

    Returns:
        np.ndarray: _description_
    """
    binning = binning or 1
    array = array[: len(array) // binning * binning]
    x_reshaped = np.reshape(array, (-1, binning))
    return np.mean(x_reshaped, axis=1)


d = so_dict.so_dict()

# Define spectra path and template to read it
try:
    d.read_from_file(sys.argv[1])
    spectra_path = d["spec_dir"]
    plot_dir = d["plots_dir"]
    plots_yaml = d["plots_yaml"]
except:
    raise ValueError("couldn't load paramfile or yaml :(")
    # spectra_path = "/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1019_carlos_150"
    # d.read_from_file(spectra_path + "/_paramfile.dict")
    # yaml_path = "python/plots_1019.yaml"

log = log.get_logger(**d)
with open(plots_yaml, "r") as f:
    plot_info: dict = yaml.safe_load(f)["plot_everything.py"]

spectra_cross_template = spectra_path + "/Dl_{}x{}_cross.dat"
spectra_auto_template = spectra_path + "/Dl_{}x{}_auto.dat"
spectra_noise_template = spectra_path + "/Dl_{}x{}_noise.dat"

# Define where and what to plot
save_path = plot_dir + "/spectra/"
os.makedirs(save_path, exist_ok=True)
save_path_cross = save_path + "cross/"
os.makedirs(save_path_cross, exist_ok=True)
save_path_cross_freqs = save_path + "cross_freqs/"
os.makedirs(save_path_cross_freqs, exist_ok=True)
save_path_cross_noises = save_path + "noises/"
os.makedirs(save_path_cross_noises, exist_ok=True)
save_path_TF = save_path + "TF/"
os.makedirs(save_path_TF, exist_ok=True)

clfile = "/global/cfs/cdirs/sobs/users/merrydup/deep56/proposal_plots/cmb.dat"
l, ps_theory = so_spectra.read_ps(clfile, spectra=spectra)
lmax = d["lmax"]
binning_file = d["binning_file"]
Dlb_theory_dumb = {
    spec: pspy_utils.naive_binning(
        l[:lmax],
        ps_theory[spec][:lmax],
        binning_file=binning_file,
        lmax=d["lmax"],
    )[1]
    for spec in ["TT", "EE", "TE"]
}

planck_mapping = {"090": "100", "150": "143", "220": "217", "280": "353"}

surveys = d["surveys"]
surveys_arrays = [
    f"{survey}_{ar}" for survey in d["surveys"] for ar in d[f"arrays_{survey}"]
]
surveys_arrays_dict = {
    survey: [f"{survey}_{ar}" for ar in d[f"arrays_{survey}"]]
    for survey in d["surveys"]
}

log.info("load spectra")
Dls_cross = {}
Dls_noise = {}
for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(surveys_arrays, r=2):
    ls, Dls_cross[f"{sv_ar1}x{sv_ar2}"] = so_spectra.read_ps(
        spectra_cross_template.format(sv_ar1, sv_ar2), spectra=spectra
    )
    try:
        ls, Dls_noise[f"{sv_ar1}x{sv_ar2}"] = so_spectra.read_ps(
            spectra_noise_template.format(sv_ar1, sv_ar2), spectra=spectra
        )
    except:
        pass

log.info(f"load cov")
Dls_error = {}
cov_type = "analytic_cov"
cov_template = d["cov_dir"] + f"/{cov_type}" + "_{}x{}_{}x{}.npy"
loaded_errors = []
for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(surveys_arrays, r=2):
    if plot_info['load_cov']:
        try:
            cov = np.load(cov_template.format(sv_ar1, sv_ar2, sv_ar1, sv_ar2))
            Dls_error[f"{sv_ar1}x{sv_ar2}"] = {
                spec: so_cov.get_sigma(
                    cov, spectra_order=spectra, n_bins=len(ls), spectrum=spec
                )
                for spec in spectra
            }
            loaded_errors.append(f"{sv_ar1}x{sv_ar2}")
        except:
                Dls_error[f"{sv_ar1}x{sv_ar2}"] = {
                    spec: np.zeros_like(ls, dtype=np.float64)
                    for spec in spectra
                }
    else:
            Dls_error[f"{sv_ar1}x{sv_ar2}"] = {
                spec: np.zeros_like(ls, dtype=np.float64)
                for spec in spectra
            }

if list(Dls_cross.keys()) == loaded_errors:
    log.info("All Covs successfully loaded")
elif loaded_errors == []:
    log.info("WARNING : no cov loaded")
else:
    for k in Dls_cross.keys():
        if k not in loaded_errors:
            log.info(f"{k} cov not loaded")

fac = ls * (ls + 1) / (2 * np.pi)
beams = {
    sv_ar: pspy_utils.naive_binning(
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[0],
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1]
        / (max(np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1])),
        d["binning_file"],
        lmax=d["lmax"],
    )[1]
    for sv_ar in surveys_arrays
}

if plot_info['survey_cross']:
    log.info(f"survey cross")
    # cross surveys cross plots
    for f in spectra:
        for survey_1, survey_2 in itertools.combinations_with_replacement(surveys, r=2):
            fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
            ax.plot(l, ps_theory[f], color="black", label="theory")
            for i, sv_ar1 in enumerate(surveys_arrays_dict[survey_1]):
                for j, sv_ar2 in enumerate(surveys_arrays_dict[survey_2]):
                    if (i > j) and (survey_1 == survey_2):
                        continue
                    ax.errorbar(
                        ls,
                        Dls_cross[f"{sv_ar1}x{sv_ar2}"][f],
                        Dls_error[f"{sv_ar1}x{sv_ar2}"][f],
                        label=f"{sv_ar1}x{sv_ar2}",
                        mfc="white",
                        marker=".",
                        lw=0.5,
                    )

            ax.set_xlabel(r"$\ell$", fontsize=18)
            ax.set_ylabel(rf"$D^{{{f}}}_\ell$", fontsize=18)
            ax.set_title(f)
            ax.set_yscale(plot_info["yscale"][f])
            ax.set_ylim(*plot_info["cross_AxB"]["ylims"][f])
            ax.set_xlim(*plot_info["cross_AxB"]["xlims"][f])
            ax.legend()

            plt.savefig(save_path_cross + f"cross_{survey_1}x{survey_2}_{f}")
            plt.close()

if plot_info['freq_cross']:
    log.info(f"cross plots per frequency")
    # cross plots per frequency
    for freq1, freq2 in itertools.combinations_with_replacement(plot_info['frequencies'], r=2):
        for sv1, sv2 in itertools.product(surveys, repeat=2):
            save_path_cross_freqs_comb = save_path_cross_freqs + f"/{sv1}_{freq1}x{sv2}_{freq2}/"
            os.makedirs(save_path_cross_freqs_comb, exist_ok=True)
            for f in spectra:
                fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
                ax.plot(
                    l,
                    ps_theory[f],
                    color="black",
                    label="theory",
                )
                for sv_ar1 in surveys_arrays_dict[sv1]:
                    for sv_ar2 in surveys_arrays_dict[sv2]:
                        if "planck" in sv_ar1:
                            cond_1 = True if planck_mapping[freq1] in sv_ar1 else False
                        else:
                            cond_1 = True if freq1 in sv_ar1 else False
                        if "planck" in sv_ar2:
                            cond_2 = True if planck_mapping[freq2] in sv_ar2 else False
                        else:
                            cond_2 = True if freq2 in sv_ar2 else False

                        if cond_1 & cond_2:
                            try:
                                ax.errorbar(
                                    ls,
                                    Dls_cross[f"{sv_ar1}x{sv_ar2}"][f],
                                    Dls_error[f"{sv_ar1}x{sv_ar2}"][f],
                                    label=f"{sv_ar1}x{sv_ar2}",
                                    mfc="white",
                                    marker=".",
                                    lw=0.5,
                                    )
                            except:
                                if freq1 != freq2:
                                    ax.errorbar(
                                        ls,
                                        Dls_cross[f"{sv_ar2}x{sv_ar1}"][f],
                                        Dls_error[f"{sv_ar2}x{sv_ar1}"][f],
                                        label=f"{sv_ar1}x{sv_ar2}",
                                        mfc="white",
                                        marker=".",
                                        lw=0.5,
                                        )

                ax.set_xlabel(r"$\ell$", fontsize=18)
                ax.set_ylabel(rf"$D^{{{f}}}_\ell$", fontsize=18)
                ax.set_title(f)
                ax.set_yscale(plot_info["yscale"][f])
                ax.set_ylim(*plot_info["cross_AxB"]["ylims"][f])
                ax.set_xlim(*plot_info["cross_AxB"]["xlims"][f])
                ax.legend()

                plt.savefig(save_path_cross_freqs_comb + f"cross_{f}_{freq1}x{freq2}")
                plt.close()

if plot_info['Cell_noise']:
    log.info(f"Cell noise plots")
    # noise plots in Cell without deconv
    Nls = {}
    rms_list = [5, 10, 15, 20, 25, 30]
    for rms in rms_list:
        ls_nls, Nls[rms] = pspy_utils.get_nlth_dict(
            rms, type="Cl", lmax=d["lmax"], spectra=spectra
        )

    # Choose a colormap and create a list of colors
    cmap = plt.get_cmap("viridis")  # try 'plasma', 'coolwarm', 'turbo', etc.
    norm = Normalize(vmin=0, vmax=len(rms_list) - 1)
    colors = [cmap(norm(i)) for i in range(len(rms_list))]

    for f in spectra_auto:
        fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
        ax.plot(l, ps_theory[f], color="black", label="theory")
        for sv_ar in surveys_arrays:
            ax.plot(
                ls,
                Dls_noise[f"{sv_ar}x{sv_ar}"][f] / fac * beams[sv_ar] ** 2,
                label=f"{sv_ar}",
            )

        for i, (rms, nls) in enumerate(Nls.items()):
            ax.plot(
                ls_nls, nls[f], label=f"{rms}", color=colors[i], lw=4, alpha=0.4, zorder=-10
            )

        ax.set_xlabel(r"$\ell$", fontsize=18)
        ax.set_ylabel(rf"$N^{{{f}}}_\ell$", fontsize=18)
        ax.set_title(f)
        ax.set_yscale("log")
        ax.set_ylim(*plot_info["noise_Cell_BxB"]["ylims"][f])
        ax.set_xlim(*plot_info["noise_Cell_BxB"]["xlims"][f])
        ax.legend(loc='right')
        plt.savefig(save_path_cross_noises + f"noise_Cl_RMS_{f}")
        plt.close()

if plot_info['Dell_noise']:
    log.info(f"Dell noise plots")
    # unbcorrelated noise plots
    for f in spectra_auto:
        for survey in surveys:
            fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
            ax.plot(l, ps_theory[f], color="black", label="theory")
            for sv_ar1 in surveys_arrays_dict[survey]:
                ax.plot(
                    ls,
                    Dls_noise[f"{sv_ar1}x{sv_ar1}"][f],
                    label=f"{sv_ar1}",
                )
            ax.set_xlabel(r"$\ell$", fontsize=18)
            ax.set_ylabel(rf"$N^{{{f}}}_\ell$", fontsize=18)
            ax.set_title(f)
            # ax.set_yscale('log')
            ax.set_ylim(*plot_info["noise_Dell_BxB"]["ylims"][f])
            ax.set_xlim(*plot_info["noise_Dell_BxB"]["xlims"][f])
            ax.legend(loc='right')
            if f == f[::-1]:
                ax.set_yscale('log')
            plt.savefig(save_path_cross_noises + f"noise_{survey}_{f}")
            plt.close()

if plot_info['Dell_corr_noise']:
    log.info(f"Dell corr noise plots")
    # correlated noise plots
    for f in spectra:
        for survey in surveys:
            fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
            ax.plot(l, ps_theory[f], color="black", label="theory")
            for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(surveys_arrays_dict[survey], r=2):
                ax.plot(
                    ls,
                    Dls_noise[f"{sv_ar1}x{sv_ar2}"][f],
                    label=f"{sv_ar1}x{sv_ar2}",
                )
            ax.set_xlabel(r"$\ell$", fontsize=18)
            ax.set_ylabel(rf"$N^{{{f}}}_\ell$", fontsize=18)
            ax.set_title(f)
            # ax.set_yscale('log')
            ax.set_ylim(*plot_info["noise_corr_Dell_BxB"]["ylims"][f])
            ax.set_xlim(*plot_info["noise_corr_Dell_BxB"]["xlims"][f])
            ax.legend(loc='right')
            if f == f[::-1]:
                ax.set_yscale('log')
            plt.savefig(save_path_cross_noises + f"noise_corr_{survey}_{f}")
            plt.close()

if plot_info['TF']:
    log.info(f"TF plots")
    # Plot Transfer function
    xlims = {
        "TT": [0, 3000],
        "EE": [0, 3000],
    }

    ylims = {
        "TT": [0.1, 2],
        "EE": [0.1, 2],
    }


    for f in ["TT", "EE"]:
        for survey_1, survey_2 in itertools.combinations_with_replacement(surveys, r=2):
            fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
            ax.plot(l, l * 0.0 + 1, color="black", label="theory", lw=0.7, ls="--")
            for i, sv_ar1 in enumerate(surveys_arrays_dict[survey_1]):
                for j, sv_ar2 in enumerate(surveys_arrays_dict[survey_2]):
                    if (i > j) and (survey_1 == survey_2):
                        continue
                    ax.plot(
                        bin_array(ls, plot_info['transfer_function']['bin_sizes'][f]),
                        bin_array(
                            Dls_cross[f"{sv_ar1}x{sv_ar2}"][f] / Dlb_theory_dumb[f],
                            plot_info['transfer_function']['bin_sizes'][f],
                        ),
                        label=f"{sv_ar1}x{sv_ar2}",
                        ls="-",
                        lw=0.8,
                        marker=".",
                        mfc="white",
                        ms=6,
                    )

            ax.set_xlabel(r"$\ell$", fontsize=18)
            ax.set_ylabel(rf"$D^{{{f}}}_\ell / D^{{{f}, th}}_\ell$", fontsize=18)
            ax.set_title(f"{f} TF")
            # ax.set_yscale(plot_info['yscale'][f])
            ax.set_xlim(*plot_info['transfer_function']['xlims'][f])
            ax.set_ylim(*plot_info['transfer_function']['ylims'][f])
            ax.legend()
            plt.savefig(save_path_TF + f"TF_{survey_1}x{survey_2}_{f}")
            plt.close()
        for freq1, freq2 in itertools.combinations_with_replacement(plot_info['frequencies'], r=2):
            for sv1, sv2 in itertools.product(surveys, repeat=2):
                fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
                ax.plot(l, l * 0.0 + 1, color="black", label="theory", lw=0.7, ls="--")
                for sv_ar1 in surveys_arrays_dict[sv1]:
                    for sv_ar2 in surveys_arrays_dict[sv2]:
                        if "planck" in sv_ar1:
                            cond_1 = True if planck_mapping[freq1] in sv_ar1 else False
                        else:
                            cond_1 = True if freq1 in sv_ar1 else False
                        if "planck" in sv_ar2:
                            cond_2 = True if planck_mapping[freq2] in sv_ar2 else False
                        else:
                            cond_2 = True if freq2 in sv_ar2 else False
                        if cond_1 & cond_2:
                            try:
                                ax.plot(
                                        bin_array(ls, plot_info['transfer_function']['bin_sizes'][f]),
                                        bin_array(
                                            Dls_cross[f"{sv_ar1}x{sv_ar2}"][f] / Dlb_theory_dumb[f],
                                            plot_info['transfer_function']['bin_sizes'][f],
                                        ),
                                        label=f"{sv_ar1}x{sv_ar2}",
                                        ls="-",
                                        lw=0.8,
                                        marker=".",
                                        mfc="white",
                                        ms=6,
                                    )
                            except:
                                if sv1 != sv2:
                                    ax.plot(
                                            bin_array(ls, plot_info['transfer_function']['bin_sizes'][f]),
                                            bin_array(
                                                Dls_cross[f"{sv_ar2}x{sv_ar1}"][f] / Dlb_theory_dumb[f],
                                                plot_info['transfer_function']['bin_sizes'][f],
                                            ),
                                            label=f"{sv_ar2}x{sv_ar1}",
                                            ls="-",
                                            lw=0.8,
                                            marker=".",
                                            mfc="white",
                                            ms=6,
                                        )
                ax.set_xlabel(r"$\ell$", fontsize=18)
                ax.set_ylabel(rf"$D^{{{f}}}_\ell / D^{{{f}, th}}_\ell$", fontsize=18)
                ax.set_title(f"{f} TF")
                ax.set_xlim(*plot_info['transfer_function']['xlims'][f])
                ax.set_ylim(*plot_info['transfer_function']['ylims'][f])
                ax.legend()
                plt.savefig(save_path_TF + f"TF_{sv1}_{freq1}x{sv2}_{freq2}_{f}")
                plt.close()
