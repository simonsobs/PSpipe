"""
This script plot the combined TT spectra
"""

from pspy import so_dict, so_spectra, so_cov, pspy_utils
from pspipe_utils import  log
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
import pspipe_utils
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.size"] = "20"
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

show_220 = False

tag = d["best_fit_tag"]
bestfit_dir = f"best_fits{tag}"
combined_spec_dir = f"combined_spectra{tag}"

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

type = d["type"]

########################################################################################
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if show_220 == True:
    case_list = ["150x150", "90x90", "90x150", "90x220", "150x220", "220x220"]
else:
    case_list = ["90x90", "150x150", "90x150", "90x220", "150x220"]

########################################################################################

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)


color_list = ["green", "red", "blue", "purple", "gray", "cyan"]

count = 0


plt.figure(figsize=(16, 12))
plt.semilogy()
plt.ylabel(r"$D^{TT}_{\ell} [\mu K^{2}] $", fontsize=40)
plt.xlabel(r"$\ell$", fontsize=40)

plt.plot(lth, Dlth["TT"], color="gray", alpha=0.6, linestyle="--")

plt.ylim(20, 7*10**3)
plt.xlim(0,8000)
for color, case in zip(color_list, case_list):
            
    lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_{case}_TT.dat", unpack=True)
    lb_ml, vec_th_ml = np.loadtxt(f"{combined_spec_dir}/bestfit_{case}_TT.dat", unpack=True)

    id = np.where(lb_ml>1500)
    
    fa, fb = case.split("x")
    
    plt.errorbar(lb_ml, vec_ml, sigma_ml , fmt=".", color=color,  label=f"{fa} GHz x {fb} GHz")
    plt.errorbar(lb_ml[id], vec_th_ml[id], color=color, fmt="--", alpha=0.7)

plt.legend(fontsize=24)
plt.savefig(f"{paper_plot_dir}/multifrequency_spectra_TT{tag}.pdf", bbox_inches="tight")
plt.clf()
plt.close()
