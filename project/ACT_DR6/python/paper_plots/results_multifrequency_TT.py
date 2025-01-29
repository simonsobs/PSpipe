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
    case_list = [ "90x90", "90x150", "150x150", "90x220", "150x220"]

########################################################################################

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

color_list = ["green", "red", "royalblue", "orange", "purple", "yellow"]

mK_to_muK = 10**3
lpow = 0.1
multipole = np.array([ 1000, 2000, 3000, 4000, 5000])
ell_tick = multipole ** lpow
ell_label = multipole


plt.figure(figsize=(16, 8))
plt.ylabel(r"$\ell^{2} D^{TT}_{\ell} [m K^{2}] $", fontsize=40)
plt.xlabel(r"$\ell$", fontsize=40)

plt.plot(lth ** lpow, Dlth["TT"] * lth ** 2 / mK_to_muK ** 2, color="gray", alpha=0.6, linestyle="--")

shift_dict = {}
shift_dict["90x90"] = -5
shift_dict["90x150"] = 5
shift_dict["150x150"] = 0
shift_dict["90x220"] = -10
shift_dict["150x220"] = 10

plt.ylim(0.3 * 10 ** 3, 2 * 10 ** 3)
plt.xlim(500 ** lpow, 4500 ** lpow)
shift=5
count=0

for color, case in zip(color_list, case_list):
            
    lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_{case}_TT.dat", unpack=True)
    lb_ml, vec_th_ml = np.loadtxt(f"{combined_spec_dir}/bestfit_{case}_TT.dat", unpack=True)

    if "220" in case:
        id = np.where(lb_ml>1500)
        alpha = 0.4
    else:
        id = np.where(lb_ml>500)
        alpha = 1
    
    fa, fb = case.split("x")
    
    plt.errorbar((lb_ml[id]+ shift_dict[case]) ** lpow, vec_ml[id] * lb_ml[id] **2 / mK_to_muK **2, sigma_ml[id] * lb_ml[id] **2 / mK_to_muK **2,
                 fmt=".", color=color,  label=f"{fa} GHz x {fb} GHz", alpha=alpha, mfc='w')
    plt.errorbar((lb_ml[id]) ** lpow, vec_th_ml[id] * lb_ml[id]  **2 / mK_to_muK ** 2,
                  color=color, fmt="--", alpha=0.7*alpha)

    count += 1
plt.xticks(ticks=ell_tick, labels=ell_label,fontsize=20)
plt.legend(fontsize=20, loc=(0.55,0.6))
plt.savefig(f"{paper_plot_dir}/multifrequency_spectra_TT{tag}.pdf", bbox_inches="tight")
plt.clf()
plt.close()
