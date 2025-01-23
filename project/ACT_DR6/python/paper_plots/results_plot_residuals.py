"""
This script plot the residuals with respect to the LCDM model specified in the dictionnary file
"""

from pspy import so_dict, so_spectra, pspy_utils
from pspipe_utils import  log, best_fits, external_data, covariance
import numpy as np
import pylab as plt
import sys, os
from matplotlib import rcParams
import pspipe_utils
import scipy.stats as ss

rcParams["font.family"] = "serif"
rcParams["font.size"] = "18"
rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18
rcParams["axes.labelsize"] = 18
rcParams["axes.titlesize"] = 18

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]
binning_file = d["binning_file"]
lmax = d["lmax"]

combined_spec_dir = f"combined_spectra{tag}"
bestfit_dir = f"best_fits{tag}"

run_name = {}
run_name["_paper"] = "ACT"
run_name["_paper_PACT"] = "PACT"
run_name["_Planck"] = "Planck"
run_name["_Planck_LB"] = "Planck LB"

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

type = d["type"]

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")

########################################################################################
selected_spectra_list = [["TT"], ["EE"], ["TE", "ET"]]
########################################################################################

ylim = {}
ylim["TT"] = [10, 7000]
ylim["TE"] = [-105000, 75000]
ylim["EE"] = [0, 45]

ylim_res = {}
ylim_res["TT"] =  [-120000, 120000]
ylim_res["TE"] = [-15000, 15000]
ylim_res["EE"] = [-4, 4]

fac = {}
fac["TT"] = 0
fac["TE"] = 1
fac["EE"] = 0
res_fac = {}
res_fac["TT"] = 1
res_fac["TE"] = 1
res_fac["EE"] = 0

y_ticks_res = {}
y_ticks_res["TT"] = [-100000, -50000, 0 , 50000, 100000]
y_ticks_res["EE"] =  [-3, -2, -1, 0, 1, 2, 3]
y_ticks_res["TE"] = [-10000, -5000, 0, 5000, 10000]

    
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)
Dlb_th = {}
for spectrum in spectra:
    lb_th, Dlb_th[spectrum] = pspy_utils.naive_binning(lth, Dlth[spectrum], binning_file, lmax)


l_planck, ps_planck_b, sigma_planck, cov_planck = external_data.get_planck_cmb_only_data()
l_b, ps_th_plank = external_data.bin_ala_planck_cmb_only(lth, Dlth)

rebin_fac = 2

for spec_select in selected_spectra_list:
    s_name = spec_select[0]
    
    lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_all_{s_name}_cmb_only.dat", unpack=True)
    cov_ml = np.load(f"{combined_spec_dir}/cov_all_{s_name}.npy")
    inv_cov_ml = np.linalg.inv(cov_ml)
    
    l_p = l_planck[s_name]
    f_p = l_p * (l_p + 1) / (2 * np.pi)
    Dl_p = ps_planck_b[s_name] * f_p
    sigma_p = sigma_planck[s_name] * f_p
    cov_p = cov_planck[s_name+s_name] * np.outer(f_p, f_p)
    Dl_p_th = ps_th_plank[s_name] * f_p
    
    l_p_rebin, Dl_p_rebin, cov_rebin = covariance.rebin_spectrum_with_cov(l_p, Dl_p, cov_p, rebin_fac=rebin_fac)
    _, Dl_p_th_rebin, _ = covariance.rebin_spectrum_with_cov(l_p, Dl_p_th, cov_p, rebin_fac=rebin_fac)

   
    id = np.where(lb_th >= lb_ml[0])
    res = (vec_ml -  Dlb_th[s_name][id])
    chi2 = res @ inv_cov_ml @ res
    ndof = len(lb_ml)
    pte = 1 - ss.chi2(ndof).cdf(chi2)
    
    
    res_p = (Dl_p - Dl_p_th)
    chi_2_p = res_p @ np.linalg.inv(cov_p) @ res_p
    ndof_p = len(Dl_p)
    pte_p = 1 - ss.chi2(ndof_p).cdf(chi_2_p)

    res_p_rebin = (Dl_p_rebin - Dl_p_th_rebin)
    chi_2_p_rebin = res_p_rebin @ np.linalg.inv(cov_rebin) @ res_p_rebin
    ndof_p_rebin = len(Dl_p_rebin)
    pte_p_rebin = 1 - ss.chi2(ndof_p_rebin).cdf(chi_2_p_rebin)
    sigma_p_rebin = np.sqrt(cov_rebin.diagonal())


    print(f"{s_name}, ACT", pte)
    print(f"{s_name}, Planck", pte_p) )
    print(f"{s_name}, Planck (rebined)", pte_p_rebin) )

    plt.figure(figsize=(20, 9))
    plt.subplot(2,1,1)
    if s_name == "TT": plt.semilogy()
    plt.ylim(ylim[s_name])
    plt.errorbar(lb_ml, vec_ml *  lb_ml ** fac[s_name], sigma_ml * lb_ml ** fac[s_name] , fmt=".", color="royalblue", label="ACT")
    plt.errorbar(l_p_rebin, Dl_p_rebin * l_p_rebin ** fac[s_name], sigma_p_rebin * l_p_rebin ** fac[s_name], fmt=".", color="darkorange", alpha=0.6, label="Planck")
    plt.plot(lth, Dlth[s_name] * lth ** fac[s_name], color="gray", alpha=1, label=r" %s $\Lambda$CDM" % run_name[tag])

    if fac[s_name] == 0:
        plt.ylabel(r"$D^{%s}_{\ell}$" % s_name, fontsize=22)
    if fac[s_name] == 1:
        plt.ylabel(r"$\ell D^{%s}_{\ell}$" % s_name, fontsize=22)
    if fac[s_name] > 1:
        plt.ylabel(r"$\ell^{%s}D^{%s}_{\ell}$" % (fac[s_name], s_name), fontsize=22)

    plt.xlim(0,4000)
    plt.legend(fontsize=16)
    plt.xticks([])

    plt.subplot(2,1,2)
    plt.xlabel(r"$\ell$", fontsize=22)
    
    if res_fac[s_name] == 0:
        plt.ylabel(r"$(D^{%s}_{\ell} - D^{%s, th}_{\ell, \rm %s}) $" % ( s_name, s_name, run_name[tag]), fontsize=22)
    if res_fac[s_name] == 1:
        plt.ylabel(r"$\ell (D^{%s}_{\ell} - D^{%s, th}_{\ell, \rm %s}) $" % (s_name, s_name, run_name[tag]), fontsize=22)
    if res_fac[s_name] > 1:
        plt.ylabel(r"$\ell^{%s} (D^{%s}_{\ell} - D^{%s, th}_{\ell, \rm %s}) $" % (res_fac[s_name], s_name, s_name, run_name[tag]), fontsize=22)


    plt.errorbar(lb_ml, res  *  lb_ml ** res_fac[s_name], sigma_ml  *  lb_ml ** res_fac[s_name],
                 label=r"ACT", fmt=".", color="royalblue", elinewidth=2)
    plt.errorbar(l_p_rebin, res_p_rebin  *  l_p_rebin ** res_fac[s_name], sigma_p_rebin  *  l_p_rebin ** res_fac[s_name],
                 label=r"Planck", alpha=0.6, fmt=".", color="darkorange", elinewidth=2)
    
    plt.plot(lb_th, lb_th * 0, color="gray")
    plt.xlim(0,4000)
    plt.ylim(ylim_res[s_name])
    plt.yticks(ticks=y_ticks_res[s_name], labels=y_ticks_res[s_name])
    plt.legend(fontsize=16)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"{paper_plot_dir}/residal_vs_best_fit_cmb_{s_name}{tag}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()

