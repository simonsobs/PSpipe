"""
This script compute best fit from theory and fg power spectra.
It uses camb and the foreground model of mflike based on fgspectra
"""
import matplotlib
matplotlib.use("Agg")
import sys

import numpy as np
import pylab as plt
from pspy import pspy_utils, so_dict, so_spectra
from pspipe_utils import pspipe_list, best_fits
from itertools import product

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# first let's get a list of all frequency we plan to study
surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

freq_list = pspipe_list.get_freq_list(d)
# let's create the directories to write best fit to disk and for plotting purpose
bestfit_dir = "best_fits"
plot_dir = "plots/best_fits/"

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)

cosmo_params = d["cosmo_params"]
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500)

f_name = f"{bestfit_dir}/cmb.dat"
so_spectra.write_ps(f_name, l_th, ps_dict, type, spectra=spectra)


fg_norm = d["fg_norm"]
fg_params = d["fg_params"]
fg_components = d["fg_components"]
fg_dict = best_fits.get_foreground_dict(l_th, freq_list, fg_components, fg_params, fg_norm)
fg= {}
for freq1 in freq_list:
    for freq2 in freq_list:
        fg[freq1, freq2] = {}
        for spec in spectra:
            fg[freq1,freq2][spec] = fg_dict[spec.lower(), "all", freq1, freq2]
        so_spectra.write_ps(f"{bestfit_dir}/fg_{freq1}x{freq2}.dat", l_th, fg[freq1,freq2], type, spectra=spectra)

for spec in spectra:
    plt.figure(figsize=(12, 12))
    for freq1 in freq_list:
        for freq2 in freq_list:
            name = f"{freq1}x{freq2}_{spec}"
            cl_th_and_fg = ps_dict[spec]

            if spec == "TT":
                plt.semilogy()

            if spec.lower() in d["fg_components"].keys():
                fg = fg_dict[spec.lower(), "all", freq1, freq2]
            else:
                fg = fg_dict[spec.lower()[::-1], "all", freq1, freq2]

            cl_th_and_fg = cl_th_and_fg + fg

            plt.plot(l_th, cl_th_and_fg, label= f"{freq1} x {freq2}")
    plt.legend()
    plt.savefig(f"{plot_dir}/best_fit_{spec}.png")
    plt.clf()
    plt.close()

nfreq = len(freq_list)

fg_components["tt"].remove("tSZ_and_CIB")
for comp in ["tSZ", "cibc", "tSZxCIB"]:
    fg_components["tt"].append(comp)

for mode in ["tt", "te", "ee"]:
    fig, axes = plt.subplots(nfreq, nfreq, sharex = True, sharey = True, figsize = (10, 10))
    for i, cross in enumerate(product(freq_list, freq_list)):
        f0, f1 = cross
        f0, f1 = int(f0), int(f1)
        idx = (i % nfreq, i // nfreq)
        ax = axes[idx]

        if idx in zip(*np.triu_indices(nfreq, k=1)) and mode != "te":
            fig.delaxes(ax)
            continue

        for comp in fg_components[mode]:
            ax.plot(l_th, fg_dict[mode, comp, f0, f1])
        ax.plot(l_th, fg_dict[mode, "all", f0, f1], color = "k")
        ax.plot(l_th, ps_dict[mode.upper()], color = "gray")
        ax.legend([], title="{}x{} GHz".format(*cross))
        if mode == "tt":
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 1e4)
        if mode == "ee":
            ax.set_yscale("log")
    for i in range(nfreq):
        axes[-1, i].set_xlabel(r"$\ell$")
        axes[i, 0].set_ylabel(r"$D_\ell$")
    fig.legend(fg_components[mode] + ["all"], title=mode.upper(), bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/foregrounds_all_comps_{mode}.png", dpi = 300)
    plt.close()
