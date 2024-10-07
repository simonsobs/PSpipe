"""
This script plot the combined dr6 spectra together with planck
"""

from pspy import so_dict, so_spectra, pspy_utils
from pspipe_utils import  log, best_fits
import numpy as np
import pylab as plt
import sys, os
from matplotlib import rcParams
import pspipe_utils


rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

combined_spec_dir = "combined_spectra"
bestfit_dir = "best_fits"

plot_dir = "plots/combined_spectra/"
pspy_utils.create_directory(plot_dir)

type = d["type"]

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")

########################################################################################
selected_spectra_list = [["TE", "ET"], ["EE"]]
########################################################################################

ylim = {}

ylim["TE"] = [0, 60000]
ylim["TE"] = [-105000, 75000]
ylim["EE"] = [0, 45]
fac = {}
fac["TE"] = 1
fac["EE"] = 0
fac["TT"] = 0

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

for spec_select in selected_spectra_list:
    s_name = spec_select[0]
    
    lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_all_{s_name}.dat", unpack=True)
    lb_ml, vec_th_ml = np.loadtxt(f"{combined_spec_dir}/bestfit_all_{s_name}.dat", unpack=True)
    cov_ml = np.load(f"{combined_spec_dir}/cov_all_{s_name}.npy")

    plt.figure(figsize=(15, 10))
    lp, Dlp, sigmap, _, _ = np.loadtxt(f"{planck_data_path}/COM_PowerSpect_CMB-{s_name}-binned_R3.02.txt", unpack=True)
    if s_name == "TT": plt.semilogy()
    
    plt.xlim(0,4000)
    plt.ylim(ylim[s_name])
    plt.errorbar(lp, Dlp * lp ** fac[s_name], sigmap * lp ** fac[s_name], fmt=".", color="royalblue", markersize=2, alpha=0.6, label="Planck PR3")
    plt.errorbar(lb_ml, vec_ml *  lb_ml ** fac[s_name], sigma_ml * lb_ml ** fac[s_name] , fmt=".", color="red", markersize=2, label="ACT DR6")
    plt.plot(lth, Dlth[s_name] * lth ** fac[s_name], color="gray", alpha=0.4)
    plt.xlabel(r"$\ell$", fontsize=30)
    
    if fac[s_name] == 0:
        plt.ylabel(r"$D_{\ell}$", fontsize=30)
    if fac[s_name] == 1:
        plt.ylabel(r"$\ell D_{\ell}$", fontsize=30)
    if fac[s_name] > 1:
        plt.ylabel(r"$\ell^{%s}D_{\ell}$" % fac[s_name], fontsize=30)

    plt.legend(fontsize=25)
    plt.savefig(f"{plot_dir}/all_spectra_{s_name}_with_planck.png", bbox_inches="tight")
    plt.clf()
    plt.close()
