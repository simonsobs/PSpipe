"""
This script plot the combined DR6 power spectra 
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import pspipe_list, log, best_fits
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.size"] = "40"
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]

combined_spec_dir = f"combined_spectra{tag}"
bestfit_dir = f"best_fits{tag}"
plot_dir = f"plots/combined_spectra{tag}/"
pspy_utils.create_directory(plot_dir)

type = d["type"]

########################################################################################

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
selected_spectra_list = [["TT"], ["TE", "ET"], ["TB", "BT"], ["EB", "BE"], ["EE"], ["BB"]]

case_list = ["all", "150x150", "90x90", "90x150", "90x220", "150x220", "220x220"]


ylim = {}
ylim["TT"] = [20, 6500]
ylim["TE"] = [-150, 150]
ylim["TB"] = [-10, 10]
ylim["EE"] = [-5, 45]
ylim["EB"] = [-1, 1]
ylim["BB"] = [-1, 1]

ylim_res = {}
ylim_res["TT"] = [-40, 40]
ylim_res["TE"] = [-10, 10]
ylim_res["TB"] = [-10, 10]
ylim_res["EE"] = [-2, 2]
ylim_res["EB"] = [-1, 1]
ylim_res["BB"] = [-1, 1]
########################################################################################

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

for spec_select in selected_spectra_list:
    s_name = spec_select[0]
    for case in case_list:
                
        if ("220" in case) & (s_name != "TT"): continue
        
        lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_{case}_{s_name}.dat", unpack=True)
        lb_ml, vec_th_ml = np.loadtxt(f"{combined_spec_dir}/bestfit_{case}_{s_name}.dat", unpack=True)
        cov_ml = np.load(f"{combined_spec_dir}/cov_{case}_{s_name}.npy")
        corr_ml = so_cov.cov2corr(cov_ml, remove_diag=True)

        chi2 = (vec_ml - vec_th_ml) @ np.linalg.inv(cov_ml) @ (vec_ml - vec_th_ml)
        ndof = len(lb_ml)
        pte = 1 - ss.chi2(ndof).cdf(chi2)

  
        plt.figure(figsize=(15, 8))
        plt.suptitle(f"{s_name} {case}", fontsize=22)
        plt.subplot(2,2,1)

        if (s_name == "TT"): plt.semilogy()

        plt.plot(lth, Dlth[s_name], color="gray", alpha=0.6, linestyle="--")
        plt.errorbar(lb_ml, vec_ml, sigma_ml, fmt=".")
        plt.errorbar(lb_ml, vec_th_ml)
        plt.xlabel(r"$\ell$", fontsize=19)
        plt.ylabel(r"$D_{\ell} [\mu K^{2}] $", fontsize=19)

        plt.ylim(ylim[s_name])

        if s_name != "TT":
            plt.xlim(0,4000)
        plt.subplot(2,2,2)
        plt.title("Correlation matrix")
        plt.imshow(corr_ml, cmap="seismic")
        plt.xticks(np.arange(len(lb_ml))[::3], lb_ml[::3], rotation=90)
        plt.yticks(np.arange(len(lb_ml))[::3], lb_ml[::3])
        plt.tight_layout()
        plt.colorbar()
        plt.subplot(2,2,3)
        plt.errorbar(lb_ml, (vec_ml - vec_th_ml) , sigma_ml , label=r"$\chi^{2}$=%.2f, pte = %.4f" % (chi2, pte), fmt=".")
        plt.plot(lb_ml, lb_ml * 0 )
        plt.plot(lth, Dlth[s_name] * 0, color="gray", alpha=0.6, linestyle="--")
        plt.xlabel(r"$\ell$", fontsize=19)
        plt.ylabel(r"$D_{\ell} - D^{\rm th}_{\ell} [\mu K^{2}]$", fontsize=19)
        plt.ylim(ylim_res[s_name])
        if s_name != "TT":
            plt.xlim(0,4000)
        plt.legend(fontsize=16)
        plt.savefig(f"{plot_dir}/spectra_{case}_{s_name}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
