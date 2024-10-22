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

rcParams["font.family"] = "serif"
rcParams["font.size"] = "40"
rcParams["xtick.labelsize"] = 40
rcParams["ytick.labelsize"] = 40
rcParams["axes.labelsize"] = 40
rcParams["axes.titlesize"] = 40

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]
combined_spec_dir = f"combined_spectra{tag}"
bestfit_dir = f"best_fits{tag}"

plot_dir = f"plots/combined_spectra{tag}/"
pspy_utils.create_directory(plot_dir)

type = d["type"]

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")

########################################################################################
selected_spectra_list = [["EE"], ["TE", "ET"]]
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


plt.figure(figsize=(40, 40))
count = 1
for spec_select in selected_spectra_list:
    s_name = spec_select[0]
    
    lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_all_{s_name}_cmb_only.dat", unpack=True)
    cov_ml = np.load(f"{combined_spec_dir}/cov_all_{s_name}.npy")

    lp, Dlp, sigmap, _, _ = np.loadtxt(f"{planck_data_path}/COM_PowerSpect_CMB-{s_name}-binned_R3.02.txt", unpack=True)
    if s_name == "TT": plt.semilogy()
    
    plt.subplot(2,1,count)
    plt.xlim(0,4000)
    plt.ylim(ylim[s_name])
    plt.errorbar(lb_ml, vec_ml *  lb_ml ** fac[s_name], sigma_ml * lb_ml ** fac[s_name] , fmt="o", color="royalblue", label="ACT")
    plt.errorbar(lp, Dlp * lp ** fac[s_name], sigmap * lp ** fac[s_name], fmt="o", color="darkorange", alpha=1, label="Planck")
    plt.plot(lth, Dlth[s_name] * lth ** fac[s_name], color="gray", alpha=0.4)
    plt.xlabel(r"$\ell$", fontsize=70)
    
    
    
    if fac[s_name] == 0:
        plt.ylabel(r"$D^{%s}_{\ell}$" % s_name, fontsize=70)
    if fac[s_name] == 1:
        plt.ylabel(r"$\ell D^{%s}_{\ell}$" % s_name, fontsize=70)
    if fac[s_name] > 1:
        plt.ylabel(r"$\ell^{%s}D^{%s}_{\ell}$" % (fac[s_name], s_name), fontsize=50)

    if count == 1:
        plt.legend(fontsize=50)
    count += 1
plt.savefig(f"{plot_dir}/all_spectra_with_planck.png", bbox_inches="tight")
plt.clf()
plt.close()
