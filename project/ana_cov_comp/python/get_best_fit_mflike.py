description = """
This script compute best fit from theory and fg power spectra.
It uses camb and the foreground model of mflike based on fgspectra
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pylab as plt
from pspipe_utils import best_fits, log, pspipe_list
from pspy import pspy_utils, so_dict, so_spectra
from itertools import combinations_with_replacement as cwr
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

# if 'lmax_pseudocov' in paramfile, use it instead of lmax
lmax = d['lmax']
if 'lmax_pseudocov' in d:
    lmax_pseudocov = d['lmax_pseudocov']
    assert lmax_pseudocov >= lmax, f'{lmax_pseudocov=} must be >= {lmax=}'
    lmax = lmax_pseudocov

type = d["type"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# let's create the directories to write best fit to disk and for plotting purpose
bestfit_dir = d["best_fits_dir"]
plot_dir = d["plot_dir"] + "/best_fits"

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)

cosmo_params = d["cosmo_params"]
log.info(f"Computing CMB spectra with cosmological parameters: {cosmo_params}")
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 1, **d["accuracy_params"])

f_name = f"{bestfit_dir}/cmb.dat"
so_spectra.write_ps(f_name, l_th, ps_dict, type, spectra=spectra)

fg_norm = d["fg_norm"]
fg_params = d["fg_params"]
fg_components = d["fg_components"]

passbands = {}
do_bandpass_integration = d["do_bandpass_integration"]

if do_bandpass_integration:
    log.info("Doing bandpass integration")

# get the passbands over which to create signal model
narrays, sv_list, ar_list = pspipe_list.get_arrays_list(d)

for sv, ar in zip(sv_list, ar_list):

    freq_info = d[f"freq_info_{sv}_{ar}"]
    if do_bandpass_integration:
        nu_ghz, pb = np.loadtxt(freq_info["passband"]).T
    else:
        nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.])

    passbands[f"{sv}_{ar}"] = [nu_ghz, pb]

log.info("Getting foregrounds contribution")
l_th = l_th[l_th < 10_000] # models only go up to 9_999
fg_dict = best_fits.get_foreground_dict(l_th, passbands, fg_components, fg_params, fg_norm)

for sv1, ar1 in zip(sv_list, ar_list):
    for sv2, ar2 in zip(sv_list, ar_list):
        name1 = f"{sv1}_{ar1}"
        name2 = f"{sv2}_{ar2}"
        fg = {}
        for spec in spectra:
            fg[spec] = fg_dict[spec.lower(), "all", name1, name2]
        so_spectra.write_ps(f"{bestfit_dir}/fg_{name1}x{name2}.dat", l_th, fg, type, spectra=spectra)

############################################################
### after this, we are done, the rest is legacy/plotting ###
############################################################

log.info("Writing best fit spectra")
spectra_list = ([f"{a}x{b}" for a,b in cwr(passbands.keys(), 2)])
best_fit_dict = {}
for ps_name in spectra_list:
    best_fit_dict[ps_name] = {}
    name1, name2 = ps_name.split("x")
    for spec in spectra:
        fg = fg_dict[spec.lower(), "all", name1, name2]
        best_fit_dict[ps_name][spec] = ps_dict[spec][:(9_999 + 1) - 2] + fg
    so_spectra.write_ps(f"{bestfit_dir}/cmb_and_fg_{ps_name}.dat", l_th, best_fit_dict[ps_name], type, spectra=spectra)


for spec in spectra:
    plt.figure(figsize=(12, 12))
    if spec == "TT":
        plt.semilogy()
    for ps_name in spectra_list:
        plt.plot(l_th, best_fit_dict[ps_name][spec], label=ps_name)
    plt.legend()
    plt.savefig(f"{plot_dir}/best_fit_{spec}.png")
    plt.clf()
    plt.close()



fg_components["tt"].remove("tSZ_and_CIB")
for comp in ["tSZ", "cibc", "tSZxCIB"]:
    fg_components["tt"].append(comp)

for mode in ["tt", "te", "tb", "ee", "eb", "bb"]:
    fig, axes = plt.subplots(narrays, narrays, sharex=True, sharey=True, figsize=(16, 16))
    indices = np.triu_indices(narrays)[::-1]
    for i, cross in enumerate(spectra_list):
        name1, name2 = cross.split("x")
        idx = (indices[0][i], indices[1][i])
        ax = axes[idx]

        for comp in fg_components[mode]:
            ax.plot(l_th, fg_dict[mode, comp, name1, name2])
        ax.plot(l_th, fg_dict[mode, "all", name1, name2], color="k")
        ax.plot(l_th, ps_dict[mode.upper()][:(9_999 + 1) - 2], color="gray")
        ax.set_title(cross)
        if mode == "tt":
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 1e4)
        if mode == "ee":
            ax.set_yscale("log")
        if mode == "bb":
            ax.set_yscale("log")

        if (mode[0] != mode[1]) and (name1 != name2):
            ax = axes[idx[::-1]]
            for comp in fg_components[mode]:
                ax.plot(l_th, fg_dict[mode, comp, name2, name1])
            ax.plot(l_th, fg_dict[mode, "all", name2, name1])
            ax.plot(l_th, ps_dict[mode.upper()][:(9_999 + 1) - 2], color="gray")
            ax.set_title(f"{name2}x{name1}")

    if mode[0] == mode[1]:
        for idx in zip(*np.triu_indices(narrays, k=1)):
            ax = axes[idx]
            fig.delaxes(ax)

    for i in range(narrays):
        axes[-1, i].set_xlabel(r"$\ell$")
        axes[i, 0].set_ylabel(r"$D_\ell$")
    fig.legend(fg_components[mode] + ["all"], title=mode.upper(), bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/foregrounds_all_comps_{mode}.png", dpi=300)
    plt.close()