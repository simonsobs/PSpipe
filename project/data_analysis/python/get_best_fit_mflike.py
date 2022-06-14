"""
This script compute best fit from theory and fg power spectra.
It uses camb and the foreground model of mflike based on fgspectra
"""
import matplotlib
matplotlib.use("Agg")
import sys

import numpy as np
import pylab as plt
from pspy import pspy_utils, so_dict

from itertools import product

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# first let's get a list of all frequency we plan to study
surveys = d["surveys"]
lmax = d["lmax"]

freq_list = []
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        freq_list += [d["nu_eff_%s_%s" % (sv, ar)]]

# remove doublons
freq_list = list(dict.fromkeys(freq_list))

# let's create the directories to write best fit to disk and for plotting purpose
bestfit_dir = "best_fits"
plot_dir = "plots/best_fits/"

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)

# now we use camb to produce best fit power spectrum, we will use CAMB to do so with standard LCDM params
import camb
ell_min, ell_max = 2, lmax + 500
cosmo_params = d["cosmo_params"]
camb_cosmo = {k: v for k, v in cosmo_params.items() if k not in ["logA", "As"]}
camb_cosmo.update({"As": 1e-10*np.exp(cosmo_params["logA"]), "lmax": ell_max, "lens_potential_accuracy": 1})
pars = camb.set_params(**camb_cosmo)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
clth = {}
for count, spec in enumerate(["TT", "EE", "BB", "TE" ]):
    clth[spec] = powers["total"][ell_min:ell_max][:,count]
clth["ET"] = clth["TE"]
for spec in ["TB", "BT", "EB", "BE" ]:
    clth[spec] = clth["TT"] * 0

ell = np.arange(ell_min, ell_max)
np.savetxt("%s/lcdm.dat" % bestfit_dir, np.transpose([ell, clth["TT"], clth["EE"], clth["BB"], clth["TE"]]))

# we will now use mflike (and in particular the fg module) to get the best fit foreground model
# we will only include foreground in tt, note that for now only extragalactic foreground are present
from mflike import theoryforge_MFLike as th_mflike
ThFo = th_mflike.TheoryForge_MFLike()
ThFo.bandint_freqs = np.array(freq_list, dtype = "float")
ThFo.freqs = ThFo.bandint_freqs

fg_norm = d["fg_norm"]
fg_model =  {"normalisation":  fg_norm, "components": d["fg_components"]}
fg_params = d["fg_params"]

ThFo.foregrounds = fg_model
ThFo._init_foreground_model()
fg_dict = ThFo._get_foreground_model(ell=ell, freqs_order=freq_list,  **fg_params)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

for spec in spectra:
    plt.figure(figsize=(12,12))
    for freq1 in freq_list:
        for freq2 in freq_list:
            name = "%sx%s_%s" % (freq1, freq2, spec)
            cl_th_and_fg = clth[spec]

            if spec == "TT":
                plt.semilogy()
                cl_th_and_fg = cl_th_and_fg + fg_dict["tt", "all", freq1, freq2]
                np.savetxt("%s/fg_%s.dat" % (bestfit_dir, name),
                                    np.transpose([ell, fg_dict["tt", "all", freq1, freq2]]))


            np.savetxt("%s/best_fit_%s.dat" % (bestfit_dir, name),
                                np.transpose([ell, cl_th_and_fg]))

            plt.plot(ell, cl_th_and_fg, label= "%s x %s" %(freq1, freq2) )
    plt.legend()
    plt.savefig("%s/best_fit_%s.png" % (plot_dir, spec))
    plt.clf()
    plt.close()

nfreq = len(freq_list)

fg_model["components"]["tt"].remove("tSZ_and_CIB")
for comp in ["tSZ", "cibc", "tSZxCIB"]:
    fg_model["components"]["tt"].append(comp)

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

        for comp in fg_model["components"][mode]:
            ax.plot(ell, fg_dict[mode, comp, f0, f1])
        ax.plot(ell, fg_dict[mode, "all", f0, f1], color = "k")
        ax.plot(ell, clth[mode.upper()], color = "gray")
        ax.legend([], title="{}x{} GHz".format(*cross))
        if mode == "tt":
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 1e4)
        if mode == "ee":
            ax.set_yscale("log")
    for i in range(nfreq):
        axes[-1, i].set_xlabel(r"$\ell$")
        axes[i, 0].set_ylabel(r"$D_\ell$")
    fig.legend(fg_model["components"][mode] + ["all"], title=mode.upper(), bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig("%s/foregrounds_all_comps_%s.png"%(plot_dir,mode), dpi = 300)
    plt.close()
