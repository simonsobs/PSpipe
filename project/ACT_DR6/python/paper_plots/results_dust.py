"""
this script plot the result for the 353x353-143x143 Planck specra combination that is used
to infer the dust in our patch of observation
You would need to have run all codes in  https://github.com/simonsobs/PSpipe/blob/master/project/data_analysis/dust.rst
Before being able to run this script
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, log, best_fits, external_data
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from matplotlib import rcParams

rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

spectra = ["TT", "TE", "EE", "BB"]
colors = ["blue", "green", "orange", "red", "purple"]

dust_dir = "results_dust"
pspy_utils.create_directory(dust_dir)

ylim = {}
ylim["TT"] = [4000, 6400]
ylim["TE"] = [0, 400]
ylim["EE"] = [-160, 200]
ylim["BB"] = [-160, 200]

nparams = {}
nparams["TT"] = 3
nparams["TE"] = 1
nparams["EE"] = 1
nparams["BB"] = 1

yticks = {}
yticks["TT"] = [5000, 6000]
yticks["TE"] = [100, 200, 300]
yticks["EE"] = [0, 100]
yticks["BB"] = [0, 100]

plt.figure(figsize=(10,10))
plt.suptitle(r"$D^{\rm{Planck}^{*} 353x \rm{Planck}^{*} 353}_{\ell} - D^{\rm{Planck}^{*} 143x \rm{Planck}^{*} 143}_{\ell} [\mu K^{2}]$", fontsize=22)
for i, (spectrum, col) in enumerate(zip(spectra,colors)):
    plt.subplot(4, 1, i+1)
    lb, res_planck, err_res_planck, res_planck_model = np.loadtxt(f"chains/dust_from_planck353_{spectrum}/residual.dat", unpack=True)
    i_cov = np.load(f"chains/dust_from_planck353_{spectrum}/icov_residual.npy")

    diff = res_planck - res_planck_model
    chi2 = diff @ i_cov @ diff
    
    ndof = len(lb) - nparams[spectrum]
    
    pte = 1-ss.chi2(ndof).cdf(chi2)

    plt.errorbar(lb, res_planck, err_res_planck, fmt=".", color=col, label= r"$\chi^{2}$= %.2f, p = %.2f %%" % (chi2, pte * 100))
    plt.plot(lb, res_planck_model, color=col)
    if i == 3:
        plt.xlabel(r"$\ell$")
    else:
        plt.xticks([])
    plt.ylabel(spectrum)
    plt.ylim(ylim[spectrum])
    plt.legend(loc="lower left", fontsize=15)
    plt.yticks(yticks[spectrum])
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig(f"{dust_dir}/dust_summary.png", dpi=300)
plt.close()
plt.clf()


