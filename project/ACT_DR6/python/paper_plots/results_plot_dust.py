"""
This script plot the combined dr6 spectra together with planck
"""

from pspy import so_dict, so_spectra, pspy_utils
from pspipe_utils import  log, best_fits
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from matplotlib import rcParams
import pspipe_utils
import scipy.stats as ss

labelsize = 14
fontsize = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)


tag = d["best_fit_tag"]
binning_file = d["binning_file"]
lmax = d["lmax"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

bestfit_dir = f"best_fits{tag}"

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)
spec_list = ["dr6_pa6_f090xdr6_pa6_f090", "dr6_pa6_f090xdr6_pa6_f150", "dr6_pa6_f150xdr6_pa6_f150"]
color_list = ["red", "green", "blue"]
name_list = ["90 GHz x 90 GHz", "90 GHz x 150 GHz", "150 GHz x 150 GHz"]
linestyle_list = ["-", "--"]


fig = plt.figure(figsize=(8, 8), dpi=100)
plt.semilogy()
count = 0
for linestyle, spectrum in zip(linestyle_list, ["EE", "BB"]):

    plt.plot(lth, Dlth[spectrum], color="black", linestyle=linestyle, linewidth=2)
    for spec, color, name in zip(spec_list, color_list, name_list):
        l, fg = np.loadtxt(f"{bestfit_dir}/components/{spectrum.lower()}_dust_{spec}.dat", unpack=True)
        
        plt.plot(lth, fg, color=color, linestyle=linestyle, label=f"{spectrum} {name}", alpha=0.7)
        
   # if count == 0:
    count+=1
    
plt.legend(fontsize=16, frameon=False)
plt.ylim(3*10**-3, 60)
plt.xlim(0, 6000)
plt.xlabel(r"$\ell$", fontsize=25)
plt.ylabel(r"$ D_{\ell} \ [\mu \rm K^{2}]$", fontsize=25)
plt.tick_params(labelsize=20)
plt.savefig(f"{paper_plot_dir}/dust{tag}.pdf", bbox_inches='tight')
plt.clf()
plt.close()

fig = plt.figure(figsize=(8, 8), dpi=100)


ylim = [10**-1, 100]
yticks_loc = [1, 10, 100]
yticks_name = ["1%", "10%", "100%"]


plt.semilogy()
plt.hlines(1, 0, 10000, color="gray", alpha=0.6, linestyle=":")
plt.hlines(10, 0, 10000, color="gray", alpha=0.6, linestyle=":")

count = 0
for linestyle, spectrum in zip(linestyle_list, ["EE", "BB"]):

    for spec, color, name in zip(spec_list, color_list, name_list):
        l, fg = np.loadtxt(f"{bestfit_dir}/components/{spectrum.lower()}_dust_{spec}.dat", unpack=True)
        
        plt.plot(lth, fg/(Dlth[spectrum]+fg) * 100, color=color, linestyle=linestyle, label=name, alpha=0.7)
        
    if count == 0:
        plt.legend(fontsize=labelsize)
    count+=1
plt.ylim(ylim)
plt.yticks(yticks_loc, yticks_name)

plt.xlim(0, 6000)
plt.xlabel(r"$\ell$", fontsize=fontsize)
plt.ylabel(r"$ D^{\rm dust}_{\ell} / (D^{\rm CMB}_{\ell}  + D^{\rm dust}_{\ell} )$", fontsize=fontsize)
plt.tick_params(labelsize=labelsize)
plt.savefig(f"{paper_plot_dir}/relative_contribution_dust_cmb{tag}.pdf", bbox_inches='tight')
plt.clf()
plt.close()
