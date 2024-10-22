"""
This script plot the combined TT spectra
"""

from pspy import so_dict, so_spectra, so_cov, pspy_utils
from pspipe_utils import  log
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from matplotlib import rcParams, gridspec

rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

show_220 = False

tag = d["best_fit_tag"]
bestfit_dir = f"best_fits{tag}"
combined_spec_dir = f"combined_spectra{tag}"
plot_dir = f"plots/combined_spectra{tag}/"
pspy_utils.create_directory(plot_dir)

type = d["type"]

########################################################################################
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if show_220 == True:
    case_list = ["150x150", "90x90", "90x150", "90x220", "150x220", "220x220"]
else:
    case_list = ["150x150", "90x90", "90x150", "90x220", "150x220"]

########################################################################################

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

color_list = ["blue", "green", "orange", "red", "gray", "cyan"]

count = 0
for color, case in zip(color_list, case_list):
            
    lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_{case}_TT.dat", unpack=True)
    lb_ml, vec_th_ml = np.loadtxt(f"{combined_spec_dir}/bestfit_{case}_TT.dat", unpack=True)
    cov_ml = np.load(f"{combined_spec_dir}/cov_{case}_TT.npy")
    corr_ml = so_cov.cov2corr(cov_ml, remove_diag=True)

    chi2 = (vec_ml - vec_th_ml) @ np.linalg.inv(cov_ml) @ (vec_ml - vec_th_ml)
    ndof = len(lb_ml)
    pte = 1 - ss.chi2(ndof).cdf(chi2)
    
    ax1 = plt.subplot(gs[0])
    ax1.semilogy()
    ax1.set_ylabel(r"$D_{\ell} [\mu K^{2}] $", fontsize=19)
    ax1.plot(lth, Dlth["TT"], color="black", alpha=0.6, linestyle="--")
    ax1.errorbar(lb_ml-10+count*3, vec_ml, sigma_ml, fmt=".", color=color, label=f"{case}, p = {pte:.3f}")
    ax1.errorbar(lb_ml-10+count*3, vec_th_ml, color=color)
    ax1.legend(fontsize=16)
    ax1.set_xticks([])
    ax1.set_ylim(20, 6000)
    ax1.set_xlim(0, 8000)
    ax2 = plt.subplot(gs[1])

    ax2.errorbar(lb_ml-10+count*4, (vec_ml - vec_th_ml) * lb_ml , sigma_ml * lb_ml, fmt=".", color=color)
    ax2.set_xlim(0, 8000)
    ax2.set_ylim(-100000, 100000)

    ax2.set_xlabel(r"$\ell$", fontsize=19)
    ax2.set_ylabel(r"$\ell (D_{\ell} - D^{\rm th}_{\ell}) [\mu K^{2}] $", fontsize=19)
    ax2.plot(lth, lth*0, color="black", alpha=0.6, linestyle="--")
        
    count += 1

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f"{plot_dir}/all_spectra_TT.png", bbox_inches="tight")
plt.clf()
plt.close()
