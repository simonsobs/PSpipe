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
rcParams["xtick.labelsize"] = 22
rcParams["ytick.labelsize"] = 22
rcParams["axes.labelsize"] = 30
rcParams["axes.titlesize"] = 30

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


color_list = ["royalblue", "green", "purple", "red", "gray", "cyan"]

count = 0

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")

lp, Dlp, sigmap, _, _ = np.loadtxt(f"{planck_data_path}/COM_PowerSpect_CMB-TT-binned_R3.01.txt", unpack=True)
##plt.errorbar(lp[:1500], Dlp[:1500], sigmap[:1500], color="purple", fmt=".")
id = np.where(lp<2000)
lp, Dlp, sigmap = lp[id], Dlp[id], sigmap[id]
#plt.xlim(0, 3000)
#plt.ylim(0, 2*10**9)


plt.figure(figsize=(18, 10))
plt.semilogy()
plt.errorbar(lp, Dlp, sigmap, fmt=".", color="darkorange", label="Planck")
plt.ylabel(r"$D^{TT}_{\ell} [\mu K^{2}] $", fontsize=30)
plt.xlabel(r"$\ell$", fontsize=30)

plt.plot(lth, Dlth["TT"], color="gray", alpha=0.6, linestyle="--")

plt.ylim(20, 7*10**3)
plt.xlim(0,8000)
for color, case in zip(color_list, case_list):
            
    lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_{case}_TT.dat", unpack=True)
    lb_ml, vec_th_ml = np.loadtxt(f"{combined_spec_dir}/bestfit_{case}_TT.dat", unpack=True)

    id = np.where(lb_ml>1500)
    
    plt.errorbar(lb_ml, vec_ml, sigma_ml , fmt=".", color=color,  label=f"ACT {case}")
    plt.errorbar(lb_ml[id], vec_th_ml[id], color=color)

plt.legend(fontsize=22)
plt.savefig(f"{plot_dir}/all_spectra_TT_with_planck.png", bbox_inches="tight")
plt.clf()
plt.close()

#plt.xticks([2,20,200,2000,3000,4000,5000])
#plt.legend(fontsize=16)
#plt.show()
