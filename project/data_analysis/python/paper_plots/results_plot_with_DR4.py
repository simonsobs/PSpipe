"""
The script compares the dr6 results with the DR4 (https://arxiv.org/abs/2007.07289) results
"""
from pspy import so_dict, pspy_utils
from pspipe_utils import external_data
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import scipy.stats as ss

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


tag = d["best_fit_tag"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
combined_spectra_dir = f"combined_spectra{tag}"
plot_dir = f"plots/combined_spectra{tag}/"
pspy_utils.create_directory(plot_dir)


pow_res = {}
pow_res["TT"] = 1.5
pow_res["TE"] = 0
pow_res["EE"] = -1

# Choi best fits
yp = {}
yp["90"] = 0.9853
yp["150"] = 0.9705

ylim_res = {}
ylim_res["TT", "90x90"] = [-2 * 10 ** 7, 2 * 10 ** 7]
ylim_res["TT", "90x150"] =  [- 8.66 * 10 ** 6, 7 * 10 ** 6]
ylim_res["TE", "90x90"] = [-28, 55]
ylim_res["EE", "90x90"] = [-0.01, 0.015]

ylim = {}
ylim["EE", "90x90"] = [-20, 50]
ylim["TT", "90x90"] = [20, 2000]
ylim["TT", "90x150"] = [20, 2000]
ylim["TT", "150x150"] = [20, 4000]

for spec in ["TT", "TE", "EE"]:
    fp_list, ell, cl_deep, err_deep = external_data.get_choi_spectra(spec, survey="deep", return_Dl=True)
    _, _ , cl_wide, err_wide = external_data.get_choi_spectra(spec, survey="wide", return_Dl=True)
        
    for fp in fp_list:
        print(fp)
        f0, f1 = fp.split("x")
        if fp == "150x90": continue
            
        l = ell[fp]
        if spec == "TT":
            # for TT only use deep because that's similar masking as DR6
            dr4_label = "DR4 deep"
            cl_choi = cl_deep[fp]
            err_choi = err_deep[fp]
        else:
            dr4_label = "DR4 deep + wide"

            # otherwise do an inverse variance combination
            err_choi = np.sqrt(1 / ( 1 / err_deep[fp] ** 2 + 1 / err_wide[fp] ** 2))
            cl_choi = (cl_deep[fp] / err_deep[fp] ** 2 + cl_wide[fp] / err_wide[fp] ** 2) * err_choi ** 2

        if spec == "TE":
            cl_choi, err_choi = cl_choi / yp[f1], err_choi / yp[f1]
        elif spec == "EE":
            cl_choi, err_choi = cl_choi / (yp[f0] * yp[f1]), err_choi / (yp[f0] * yp[f1])

        lb, Db, sigmab = np.loadtxt(f"{combined_spectra_dir}/Dl_{fp}_{spec}.dat", unpack=True)
    
        id = np.where((lb >= l[0]) & (lb <= l[-1]))
        lb, Db, sigmab = lb[id], Db[id], sigmab[id]
        
        id = np.where((l >= lb[0]) & (l <= lb[-1]))
        l, cl_choi, err_choi = l[id], cl_choi[id], err_choi[id]

        res = Db - cl_choi
        err_res = np.sqrt(sigmab ** 2 + err_choi ** 2)
        chi2 = np.sum(res ** 2/err_res ** 2)
        ndof = len(l)
        pte = 1 - ss.chi2(ndof).cdf(chi2)
        
        
        plt.figure(figsize=(12,8))
        plt.suptitle(f"{spec} {f0}x{f1}", fontsize=24)
        plt.subplot(2, 1, 1)
        if spec == "TT":
            plt.semilogy()
        plt.errorbar(lb, Db, sigmab, fmt=".-", label="DR6", alpha=0.5)
        plt.errorbar(l,  cl_choi, err_choi, fmt=".-", label=dr4_label, alpha=0.5)
        if spec == "EE":
            plt.errorbar(l,  l*0, fmt="--",  alpha=0.5, color="gray")

        plt.ylabel(r"$D_{\ell}$", fontsize=18)
        plt.xlabel(r"$\ell$", fontsize=18)
        try:
            ylim0 = ylim[spec, fp][0]
            ylim1 = ylim[spec, fp][1]
            plt.ylim(ylim0, ylim1)
        except:
            pass
        plt.legend(fontsize=15)
        plt.subplot(2, 1, 2)
        plt.ylabel(r"$ell^{%s} (D^{DR6}_{\ell} - D^{DR4}_{\ell}) \ $" % pow_res[spec], fontsize=15)
        plt.xlabel(r"$\ell$", fontsize=18)
        plt.errorbar(l,  res * l ** pow_res[spec], err_res * l ** pow_res[spec], fmt=".", label=r"$\chi^{2}$/Dof= %.02f/%d, pte:%0.3f"% (chi2, ndof, pte))
        try:
            ylim0 = ylim_res[spec, fp][0]
            ylim1 = ylim_res[spec, fp][1]
            plt.ylim(ylim0, ylim1)
        except:
            pass
        plt.plot(l, l * 0, color="gray")
        plt.legend(fontsize=15)
        plt.savefig(f"{plot_dir}/dr6_choi_{spec}_{fp}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
