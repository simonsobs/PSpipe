"""
Plots all kinds of spectra of combination A x B
"""

from pspy import so_spectra, pspy_utils, so_cov, so_map, so_window, so_dict
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
    spectra_path = d['spec_dir']
    plot_dir = d['plots_dir']
    yaml_path = d['yaml_dir']
except:
    print("")
    # spectra_path = "/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1019_carlos_150"
    # d.read_from_file(spectra_path + "/_paramfile.dict")
    # yaml_path = "python/plots_1019.yaml"

with open(yaml_path + '/post_spectra.yaml', "r") as f:
    plot_info: dict = yaml.safe_load(f)['plot_everything.py']

spectra_cross_template = spectra_path + "/Dl_{}x{}_cross.dat"
spectra_auto_template = spectra_path + "/Dl_{}x{}_auto.dat"
spectra_noise_template = spectra_path + "/Dl_{}x{}_noise.dat"

survey_B = "lat_iso"
arrays_B = d[f"arrays_{survey_B}"]
surveys_arrays_B = [f"{survey_B}_{ar}" for ar in arrays_B]

beams = {
    sv_ar: pspy_utils.naive_binning(
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[0],
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1]
        / (max(np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1])),
        d["binning_file"],
        lmax=d["lmax"],
    )[1]
    for sv_ar in surveys_arrays_B
}

# Define surveys and arrays to plot
for survey_A in ['dr6', 'planck']:
    arrays_A = d[f"arrays_{survey_A}"]
    # arrays_A = ['pa5_f090']
    surveys_arrays_A = [f"{survey_A}_{ar}" for ar in arrays_A]

    surveys = [survey_A, survey_B]
    surveys_arrays = surveys_arrays_A + surveys_arrays_B

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

    # Load spectra
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
    fac = ls * (ls + 1) / (2 * np.pi)

    clfile = "/global/cfs/cdirs/sobs/users/merrydup/deep56/proposal_plots/cmb.dat"
    l, ps_theory = so_spectra.read_ps(clfile, spectra=spectra)
    lmax = d["lmax"]
    binning_file = d["binning_file"]
    Dlb_theory_dumb = {
        spec: pspy_utils.naive_binning(
            l[:lmax],
            ps_theory[spec][:lmax] * l[:lmax] * (l[:lmax] + 1) / (2 * np.pi),
            binning_file=binning_file,
            lmax=d["lmax"],
        )[1]
        for spec in ["TT", "EE"]
    }

    # AxB cross plots
    for f in spectra:
        fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
        ax.plot(l, ps_theory[f] * l * (l + 1) / (2 * np.pi), color="black", label="theory")

        for sv_ar1 in surveys_arrays_A:
            for sv_ar2 in surveys_arrays_B:
                ax.plot(ls, Dls_cross[f"{sv_ar1}x{sv_ar2}"][f], label=f"{sv_ar1}x{sv_ar2}")

        ax.set_xlabel(r"$\ell$", fontsize=18)
        ax.set_ylabel(rf"$D^{{{f}}}_\ell$", fontsize=18)
        ax.set_title(f)
        ax.set_yscale(plot_info["yscale"][f])
        ax.set_ylim(*plot_info["cross_AxB"]["ylims"][f])
        ax.set_xlim(*plot_info["cross_AxB"]["xlims"][f])
        ax.legend()

        plt.savefig(save_path_cross + f"cross_{survey_A}x{survey_B}_{f}")
        plt.close()

    # AxB cross plots per frequency
    frequency_set_A = list(set([str(d[f"freq_info_{survey_A}_{ar}"]["freq_tag"]) for ar in arrays_A]))
    frequency_set_B = list(set([str(d[f"freq_info_{survey_A}_{ar}"]["freq_tag"]) for ar in arrays_A]))
    
    frequencies = frequency_set_A + frequency_set_B

    for freq1, freq2 in itertools.combinations_with_replacement(frequencies, r=2):
        if True in [freq1 in sv_ar for sv_ar in surveys_arrays_A]:
            if True in [freq2 in sv_ar for sv_ar in surveys_arrays_B]:
                for f in spectra:
                    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
                    ax.plot(
                        l,
                        ps_theory[f] * l * (l + 1) / (2 * np.pi),
                        color="black",
                        label="theory",
                    )

                    for sv_ar1 in surveys_arrays_A:
                        for sv_ar2 in surveys_arrays_B:
                            if ((freq1 in sv_ar1) & (freq2 in sv_ar2)) or ((freq2 in sv_ar1) & (freq1 in sv_ar2)):
                                ax.plot(
                                    ls,
                                    Dls_cross[f"{sv_ar1}x{sv_ar2}"][f],
                                    label=f"{sv_ar1}x{sv_ar2}",
                                )

                    ax.set_xlabel(r"$\ell$", fontsize=18)
                    ax.set_ylabel(rf"$D^{{{f}}}_\ell$", fontsize=18)
                    ax.set_title(f)
                    ax.set_yscale(plot_info["yscale"][f])
                    ax.set_ylim(*plot_info["cross_AxB"]["ylims"][f])
                    ax.set_xlim(*plot_info["cross_AxB"]["xlims"][f])
                    ax.legend()

                    plt.savefig(
                        save_path_cross_freqs
                        + f"cross_{survey_A}x{survey_B}_{f}_{freq1}x{freq2}"
                    )
                    plt.close()

    # BxB cross plots
    for f in spectra:
        fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
        ax.plot(l, ps_theory[f] * l * (l + 1) / (2 * np.pi), color="black", label="theory")

        for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(
            surveys_arrays_B, r=2
        ):
            ax.plot(ls, Dls_cross[f"{sv_ar1}x{sv_ar2}"][f], label=f"{sv_ar1}x{sv_ar2}")

        ax.set_xlabel(r"$\ell$", fontsize=18)
        ax.set_ylabel(rf"$D^{{{f}}}_\ell$", fontsize=18)
        ax.set_title(f)
        ax.set_yscale(plot_info["yscale"][f])
        ax.set_ylim(*plot_info["cross_AxB"]["ylims"][f])
        ax.set_xlim(*plot_info["cross_AxB"]["xlims"][f])
        ax.legend()

        plt.savefig(save_path_cross + f"cross_{survey_B}x{survey_B}_{f}")
        plt.close()

    # BxB cross plots per frequency
    frequencies = ["090", "150", "220", "280"]
    for freq1, freq2 in itertools.combinations_with_replacement(frequencies, r=2):
        if True in [freq1 in sv_ar for sv_ar in surveys_arrays]:
            if True in [freq2 in sv_ar for sv_ar in surveys_arrays]:
                for f in spectra:
                    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
                    ax.plot(
                        l,
                        ps_theory[f] * l * (l + 1) / (2 * np.pi),
                        color="black",
                        label="theory",
                    )

                    for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(
                            surveys_arrays_B, r=2
                        ):
                            if (freq1 in sv_ar1) & (freq2 in sv_ar2):
                                ax.plot(
                                    ls,
                                    Dls_cross[f"{sv_ar1}x{sv_ar2}"][f],
                                    label=f"{sv_ar1}x{sv_ar2}",
                                )

                    ax.set_xlabel(r"$\ell$", fontsize=18)
                    ax.set_ylabel(rf"$D^{{{f}}}_\ell$", fontsize=18)
                    ax.set_title(f)
                    ax.set_yscale(plot_info["yscale"][f])
                    ax.set_ylim(*plot_info["cross_AxB"]["ylims"][f])
                    ax.set_xlim(*plot_info["cross_AxB"]["xlims"][f])
                    ax.legend()

                    plt.savefig(
                        save_path_cross_freqs
                        + f"cross_{survey_B}x{survey_B}_{f}_{freq1}x{freq2}"
                    )
                    plt.close()

    # BxB noise plots
    for f in spectra_auto:
        fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
        ax.plot(l, ps_theory[f], color="black", label="theory")

        for sv_ar2 in surveys_arrays_B:
            ax.plot(ls, Dls_noise[f"{sv_ar2}x{sv_ar2}"][f] / fac, label=f"{sv_ar2}")

        ax.set_xlabel(r"$\ell$", fontsize=18)
        ax.set_ylabel(rf"$N^{{{f}}}_\ell$", fontsize=18)
        ax.set_title(f)
        ax.set_yscale("log")
        ax.set_ylim(*plot_info["noise_BxB"]["ylims"][f])
        ax.set_xlim(*plot_info["noise_BxB"]["xlims"][f])
        ax.legend()
        plt.savefig(save_path_cross_noises + f"noise_{survey_A}x{survey_B}_{f}")
        plt.close()

    # BxB correlated noise plots
    for f in spectra:
        if f != f[::-1]:
            fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
            ax.plot(l, ps_theory[f], color="black", label="theory")

            for sv_ar1 in surveys_arrays_B:
                for sv_ar2 in surveys_arrays_B:
                    try:
                        ax.plot(
                            ls,
                            Dls_noise[f"{sv_ar1}x{sv_ar2}"][f] / fac,
                            label=f"{sv_ar1}x{sv_ar2}",
                        )
                    except:
                        pass

            ax.set_xlabel(r"$\ell$", fontsize=18)
            ax.set_ylabel(rf"$N^{{{f}}}_\ell$", fontsize=18)
            ax.set_title(f)
            # ax.set_yscale('log')
            ax.set_ylim(*plot_info["noise_BxB"]["ylims"][f])
            ax.set_xlim(*plot_info["noise_BxB"]["xlims"][f])
            ax.legend()
            plt.savefig(save_path_cross_noises + f"corr_noise_{survey_A}x{survey_B}_{f}")
            plt.close()


    # BxB noise plots
    Nls = {}
    for rms in [5, 10, 15, 20, 25, 30]:
        ls_nls, Nls[rms] = pspy_utils.get_nlth_dict(
            rms, type="Cl", lmax=d["lmax"], spectra=spectra
        )

    # Choose a colormap and create a list of colors
    cmap = plt.get_cmap("viridis")  # try 'plasma', 'coolwarm', 'turbo', etc.
    norm = Normalize(vmin=0, vmax=5)
    colors = [cmap(norm(i)) for i in range(6)]

    for f in spectra_auto:
        fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
        ax.plot(l, ps_theory[f], color="black", label="theory")
        for sv_ar2 in surveys_arrays_B:
            ax.plot(
                ls,
                Dls_noise[f"{sv_ar2}x{sv_ar2}"][f] / fac * beams[sv_ar2] ** 2,
                label=f"{sv_ar2}",
            )

        for i, (rms, nls) in enumerate(Nls.items()):
            ax.plot(
                ls_nls, nls[f], label=f"{rms}", color=colors[i], lw=4, alpha=0.4, zorder=-10
            )

        ax.set_xlabel(r"$\ell$", fontsize=18)
        ax.set_ylabel(rf"$N^{{{f}}}_\ell$", fontsize=18)
        ax.set_title(f)
        ax.set_yscale("log")
        ax.set_ylim(*plot_info["noise_rms_BxB"]["ylims"][f])
        ax.set_xlim(*plot_info["noise_rms_BxB"]["xlims"][f])
        ax.legend()
        plt.savefig(save_path_cross_noises + f"noise_rms_{survey_A}x{survey_B}_{f}")
        plt.close()


    # Plot Transfer function
    xlims = {
        "TT": [0, 3000],
        "EE": [0, 3000],
    }

    ylims = {
        "TT": [0.5, 2],
        "EE": [0.5, 2],
    }

    bin_sizes = {
        "TT": 1,
        "EE": 3,
    }

    for f in ["TT", "EE"]:

        fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
        ax.plot(l, l * 0.0 + 1, color="black", label="theory", lw=0.7, ls="--")
        for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(
            surveys_arrays_A, r=2
        ):
            ax.plot(
                bin_array(ls, bin_sizes[f]),
                bin_array(
                    Dls_cross[f"{sv_ar1}x{sv_ar2}"][f] / Dlb_theory_dumb[f], bin_sizes[f]
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
        ax.set_xlim(*xlims[f])
        ax.set_ylim(*ylims[f])
        ax.legend()
        plt.savefig(save_path_TF + f"TF_{survey_A}x{survey_A}_{f}")
        plt.close()

        fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
        ax.plot(l, l * 0.0 + 1, color="black", label="theory", lw=0.7, ls="--")
        for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(
            surveys_arrays_B, r=2
        ):
            ax.plot(
                bin_array(ls, bin_sizes[f]),
                bin_array(
                    Dls_cross[f"{sv_ar1}x{sv_ar2}"][f] / Dlb_theory_dumb[f], bin_sizes[f]
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
        ax.set_xlim(*xlims[f])
        ax.set_ylim(*ylims[f])
        ax.legend()
        plt.savefig(save_path_TF + f"TF_{survey_B}_{f}")
        plt.close()

        fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
        ax.plot(l, l * 0.0 + 1, color="black", label="theory", lw=0.7, ls="--")
        for i, sv_ar1 in enumerate(surveys_arrays_A):
            for j, sv_ar2 in enumerate(surveys_arrays_B):
                if i <= j:
                    ax.plot(
                        bin_array(ls, bin_sizes[f]),
                        bin_array(
                            Dls_cross[f"{sv_ar1}x{sv_ar2}"][f] / Dlb_theory_dumb[f],
                            bin_sizes[f],
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
        ax.set_xlim(*xlims[f])
        ax.set_ylim(*ylims[f])
        ax.legend()
        plt.savefig(save_path_TF + f"TF_{survey_A}x{survey_B}_{f}")
        plt.close()
