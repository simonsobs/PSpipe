"""
This script plot the residuals with respect to the LCDM model specified in the dictionnary file
"""

from pspy import so_dict, so_spectra, pspy_utils
from pspipe_utils import  log, best_fits, external_data, covariance
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from matplotlib import rcParams
import pspipe_utils
import scipy.stats as ss

labelsize = 14
fontsize = 20

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
run_name["_paper_PACT"] = "P-ACT"
run_name["_Planck"] = "Planck"
run_name["_Planck_LB"] = "Planck LB"

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

xmax = 3000
type = d["type"]

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")

########################################################################################
selected_spectra_list = [["TT"], ["EE"], ["TE", "ET"]]
########################################################################################

ylim = {}
ylim["TT"] = [10, 7000]
ylim["TE"] = [-150,150]#None#[-105000, 75000]
ylim["EE"] = [0, 45]

ylim_res = {}
ylim_res["TT"] =  [-12, 12]
ylim_res["TE"] = None#[-15000, 15000]
ylim_res["EE"] = [-4, 4]

fac = {}
fac["TT"] = 0
fac["TE"] = 0
fac["EE"] = 0
res_fac = {}
res_fac["TT"] = 1
res_fac["TE"] = 0
res_fac["EE"] = 0

y_ticks_res = {}
y_ticks_res["TT"] = [-8, -4, 0 , 4, 8]
y_ticks_res["EE"] =  [-3, -2, -1, 0, 1, 2, 3]
y_ticks_res["TE"] = None#[-10000, -5000, 0, 5000, 10000]


divider_power_res = {}
divider_power_res["TT"] = 4
divider_power_res["TE"] = 0
divider_power_res["EE"] = 0

    
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)
Dlb_th = {}
for spectrum in spectra:
    lb_th, Dlb_th[spectrum] = pspy_utils.naive_binning(lth, Dlth[spectrum], binning_file, lmax)


l_planck, ps_planck_b, sigma_planck, cov_planck = external_data.get_planck_cmb_only_data()
l_b, ps_th_plank = external_data.bin_ala_planck_cmb_only(lth, Dlth)

rebin_fac = 2

fig = plt.figure(figsize=(12, 17), dpi=100)

for count, spec_select in enumerate(selected_spectra_list):
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


    print(f"{s_name}, ACT PTE", pte)
    print(f"{s_name}, Planck PTE", pte_p)
    print(f"{s_name}, Planck (rebined) PTE", pte_p_rebin)

    plt.subplot(6, 1, 1 + count * 2)
    if s_name == "TT": plt.semilogy()
    plt.ylim(ylim[s_name])

    # put planck TT dots on top of ACT dots, else ACT TE, EE on top
    if count == 0:
        act_zorder, planck_zorder = 1, 2
    else:
        act_zorder, planck_zorder = 2, 1

    plt.errorbar(lb_ml, vec_ml *  lb_ml ** fac[s_name], sigma_ml * lb_ml ** fac[s_name] , fmt="o", color="royalblue", label="ACT", mfc='w', markersize=3, elinewidth=1, zorder=act_zorder)
    plt.errorbar(l_p_rebin, Dl_p_rebin * l_p_rebin ** fac[s_name], sigma_p_rebin * l_p_rebin ** fac[s_name], fmt="o", color="darkorange", alpha=1, label="Planck", mfc='w', markersize=3, elinewidth=1, zorder=planck_zorder)
    plt.plot(lth, Dlth[s_name] * lth ** fac[s_name], color="gray", linewidth=0.7, label=r" %s $\Lambda$CDM" % run_name[tag], zorder=0)

    if fac[s_name] == 0:
        plt.ylabel(r"$D^{%s}_{\ell} \  [\mu \rm K^{2}]$" % s_name, fontsize=fontsize)
    if fac[s_name] == 1:
        plt.ylabel(r"$\ell D^{%s}_{\ell} \ [\mu \rm K^{2}]$" % s_name, fontsize=fontsize)
    if fac[s_name] > 1:
        plt.ylabel(r"$\ell^{%s}D^{%s}_{\ell} \ [\mu \rm K^{2}]$" % (fac[s_name], s_name), fontsize=fontsize)

    plt.xlim(0,xmax)
    if count == 0:
        plt.legend(fontsize=fontsize)
    plt.tick_params(axis='x', direction='in', labelbottom=False)
    plt.tick_params(labelsize=labelsize)

    plt.subplot(6, 1, 2 + count * 2)
    plt.xlabel(r"$\ell$", fontsize=fontsize)
    
    divider = 10 ** divider_power_res[s_name]
    
    if divider_power_res[s_name] == 0:
        divider_str = ""
    else:
        divider_str = r"10^{%s}" % divider_power_res[s_name]

    
    if res_fac[s_name] == 0:
        plt.ylabel(r"$ \Delta D^{%s}_{\ell}  \ [{%s} \mu \rm K^{2}] $" %  (s_name, divider_str), fontsize=fontsize)
    if res_fac[s_name] == 1:
        plt.ylabel(r"$ \ell \Delta D^{%s}_{\ell}  \ [{%s} \mu \rm K^{2}] $" %  (s_name, divider_str), fontsize=fontsize)
    if res_fac[s_name] > 1:
        plt.ylabel(r"$ \ell^{%s} \Delta D^{%s}_{\ell}  \ [{%s}  \mu \rm K^{2}] $" %  (res_fac[s_name], s_name, divider_str), fontsize=fontsize)

    plt.errorbar(lb_ml, res  *  lb_ml ** res_fac[s_name] / divider, sigma_ml  *  lb_ml ** res_fac[s_name] / divider,
                 label=r"ACT", fmt="o", color="royalblue", markersize=3, elinewidth=1, mfc='w', zorder=act_zorder)
    plt.errorbar(l_p_rebin, res_p_rebin  *  l_p_rebin ** res_fac[s_name] / divider, sigma_p_rebin  *  l_p_rebin ** res_fac[s_name] / divider,
                 label=r"Planck", fmt="o", color="darkorange", markersize=3, elinewidth=1, mfc='w', zorder=planck_zorder)
    plt.plot(lb_th, lb_th * 0, linewidth=0.7, color="gray", zorder=0)

    plt.xlim(0, xmax)
    plt.ylim(ylim_res[s_name])
    if count != 2:
        plt.tick_params(axis='x', direction='in', labelbottom=False)
    plt.tick_params(labelsize=labelsize)
    
    plt.yticks(ticks=y_ticks_res[s_name], labels=y_ticks_res[s_name])

plt.subplots_adjust(wspace=0, hspace=0)
fig.align_ylabels()

plt.savefig(f"{paper_plot_dir}/residal_vs_best_fit_cmb{tag}.pdf", bbox_inches="tight")
plt.clf()
plt.close()

