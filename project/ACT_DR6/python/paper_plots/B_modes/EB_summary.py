"""
For this to work you need to have created two folders results_EB_optimal and results_EB_uniform
using results_plot_pol_angle.py for both the uniform and weighted analysis.
Similarly you need to have created combined_spectra_paper_optimal and the corresponding simulations
"""

import numpy as np
import pylab as plt
import pickle
from matplotlib import rcParams
from pspy import so_spectra, pspy_utils, so_cov
from pspipe_utils import pol_angle
import scipy.stats as ss


binning_file = "BIN_ACTPOL_50_4_SC_large_bin_at_low_ell"
lmax = 8500


rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

latex_par_name = {}
latex_par_name["alpha_pa5_f090"] = r"$\alpha_{pa5 f090}$"
latex_par_name["alpha_pa6_f090"] = r"$\alpha_{pa6 f090}$"
latex_par_name["alpha_pa5_f150"] = r"$\alpha_{pa5 f150}$"
latex_par_name["alpha_pa6_f150"] = r"$\alpha_{pa6 f150}$"
latex_par_name["beta_pa5"] = r"\beta_{\rm pa5}"
latex_par_name["beta_pa6"] = r"\beta_{\rm pa6}"
latex_par_name["beta_ACT"] = r"\beta_{\rm ACT}"
latex_par_name["beta_ACT+komatsu"] = r"\beta_{\rm ACT + Komatsu}"
latex_par_name["beta_komatsu"] = r"\beta_{\rm Komatsu}"

cut_list = [ "post_unblinding", "pre_unblinding"]
name_list = [ "baseline multipole cut", "extended multipole cut"]
weight_list = ["uniform", "optimal"]

angle= {}
for my_w in weight_list:
    for my_cut in cut_list:
        with open(f"results_EB_{my_w}/angle_{my_cut}.pkl", "rb") as fp:
            angle[my_w, my_cut] = pickle.load(fp)


alpha_list = ["alpha_pa5_f090", "alpha_pa5_f150", "alpha_pa6_f090", "alpha_pa6_f150"]


x = np.linspace(0,4.5,100)

plt.figure(figsize=(12,6))
for count, alpha in enumerate(alpha_list):
    color_list = ["red", "orange", "blue", "lightblue"]
    count_all = 0
    for shift2, (name, my_cut) in enumerate(zip(name_list, cut_list)):
        for shift1, my_w in enumerate(weight_list):

            print(f"{alpha} {name} ({my_w} weighting)", angle[my_w, my_cut][alpha, "mean"], angle[my_w, my_cut][alpha, "std"]  )
            sh1 = shift1*0.1
            sh2 = shift2*0.2
            plt.errorbar(count+sh1+sh2,
                         angle[my_w, my_cut][alpha, "mean"],
                         angle[my_w, my_cut][alpha, "std"],
                         label=f"{name} ({my_w} weighting)",
                         color=color_list[count_all],
                         fmt="o")
            
            count_all += 1
    if count == 0:
        plt.legend(fontsize=12)
        
xticks_list = ["pa5 f090", "pa5 f150", "pa6 f090", "pa6 f150"]
plt.xticks([0.15,1.15,2.15,3.15], xticks_list, rotation=90, fontsize=22)
plt.ylabel(r"$\alpha$", fontsize=32)
plt.tight_layout()
plt.savefig("all_alpha.png")
plt.clf()
plt.close()



x = np.linspace(0,4.5,100)
beta_list = ["beta_pa5", "beta_pa6", "beta_ACT", "beta_ACT+komatsu"]
plt.figure(figsize=(12,8))

for count, beta in enumerate(beta_list):
    count_all = 0
    for shift2, (name, my_cut) in enumerate(zip(name_list, cut_list)):
        for shift1, my_w in enumerate(weight_list):
        
            print(f"{beta} {name} ({my_w} weighting)", angle[my_w, my_cut][beta, "mean"], angle[my_w, my_cut][beta, "std"]  )

            sh1 = shift1*0.1
            sh2 = shift2*0.2
            plt.errorbar(count+sh1+sh2,
                        angle[my_w, my_cut][beta, "mean"],
                        angle[my_w, my_cut][beta, "std"],
                        label=f"{name} ({my_w} weighting)",
                        color=color_list[count_all],
                        fmt="o")
                        
            count_all += 1

    if count == 0:
        plt.legend(fontsize=12)

xticks_list = ["pa5", "pa6", "ACT", "ACT+Planck"]
plt.xticks([0.15,1.15,2.15,3.15], xticks_list, rotation=90, fontsize=22)
plt.ylabel(r"$\beta$", fontsize=32)
plt.tight_layout()
plt.savefig("all_beta.png")
plt.clf()
plt.close()

beta_ACT =  angle["optimal", "post_unblinding"]["beta_ACT", "mean"]
print(beta_ACT)


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

for spec in ["EB", "TB"]:
    l, ps_th = so_spectra.read_ps("cmb.dat", spectra=spectra)
    l, psth_rot = pol_angle.rot_theory_spectrum(l, ps_th, beta_ACT, beta_ACT)

    lb, ps_th_b = pspy_utils.naive_binning(l, psth_rot[spec], binning_file, lmax)
    
    lb_dat, ps, error = np.loadtxt(f"combined_spectra_paper_optimal/Dl_all_{spec}.dat", unpack=True)
    
    id = np.where(lb>=lb_dat[0])
    
    lb, ps_th_b = lb[id], ps_th_b[id]

    cov = np.load(f"combined_spectra_paper_optimal/cov_all_{spec}.npy")

    chi2 = (ps - ps_th_b) @ np.linalg.inv(cov) @ (ps - ps_th_b)
    chi2_null = (ps) @ np.linalg.inv(cov) @ (ps)

    pte = 1 - ss.chi2(len(lb)).cdf(chi2)
    pte_null = 1 - ss.chi2(len(lb)).cdf(chi2_null)

    plt.figure(figsize=(12,6))
    plt.errorbar(l, ps_th["TT"]*0, label=f"pte (null): {pte_null*100:.5f}  %", color="gray")
    plt.errorbar(l, psth_rot[spec], label=f"pte (model): {pte*100:.2f}  %", color="red")
    plt.errorbar(lb_dat, ps, error, fmt="o", color="royalblue")
    plt.xlabel(r"$\ell$", fontsize=25)
    plt.ylabel(r"$D^{%s}_{\ell}$" % spec, fontsize=25)
    plt.legend(fontsize=14)
    if spec == "EB":
        plt.ylim(-0.3,0.6)
    if spec == "TB":
        plt.ylim(-4,6)

    plt.xlim(0,3500)
    plt.savefig(f"{spec}.png")
    plt.clf()
    plt.close()

for spec in ["EB", "TB"]:
    l, ps_th = so_spectra.read_ps("cmb.dat", spectra=spectra)
    l, psth_rot = pol_angle.rot_theory_spectrum(l, ps_th, beta_ACT, beta_ACT)

    for count, freq_pair in enumerate(["90x90", "90x150", "150x150"]):
        lb, ps_th_b = pspy_utils.naive_binning(l, psth_rot[spec], binning_file, lmax)
        lb_dat, ps, error = np.loadtxt(f"combined_spectra_paper_optimal/Dl_{freq_pair}_{spec}.dat", unpack=True)
        id = np.where(lb>=lb_dat[0])
    
        lb, ps_th_b = lb[id], ps_th_b[id]

        cov = np.load(f"combined_spectra_paper_optimal/cov_{freq_pair}_{spec}.npy")

        chi2 = (ps - ps_th_b) @ np.linalg.inv(cov) @ (ps - ps_th_b)
        chi2_null = (ps) @ np.linalg.inv(cov) @ (ps)

        pte = 1 - ss.chi2(len(lb)).cdf(chi2)
        pte_null = 1 - ss.chi2(len(lb)).cdf(chi2_null)
        
        plt.figure(figsize=(12,6))
        plt.errorbar(l, ps_th["TT"]*0, label=f"pte (null): {pte_null*100:.5f}  %", color="gray")
        plt.errorbar(l, psth_rot[spec], label=f"pte (model): {pte*100:.2f}  %", color="red")
        plt.errorbar(lb_dat, ps, error, fmt=".")
    
        plt.xlabel(r"$\ell$", fontsize=25)
        plt.ylabel(r"$D^{%s}_{\ell}$ (%s)" % (spec, freq_pair), fontsize=25)
    
        plt.legend(fontsize=14)
        if spec == "EB":
            plt.ylim(-0.3,0.6)
        if spec == "TB":
            plt.ylim(-4,6)

        plt.show()
