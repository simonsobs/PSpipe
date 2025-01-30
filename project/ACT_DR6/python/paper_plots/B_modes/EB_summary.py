import numpy as np
import pylab as plt
import pickle
from matplotlib import rcParams


rcParams["xtick.labelsize"] = 30
rcParams["ytick.labelsize"] = 30
rcParams["axes.labelsize"] = 25
rcParams["axes.titlesize"] = 25

latex_par_name = {}
latex_par_name["alpha_pa5_f090"] = r"$\psi_{pa5 f090}$"
latex_par_name["alpha_pa6_f090"] = r"$\psi_{pa6 f090}$"
latex_par_name["alpha_pa5_f150"] = r"$\psi_{pa5 f150}$"
latex_par_name["alpha_pa6_f150"] = r"$\psi_{pa6 f150}$"
latex_par_name["beta_pa5"] = r"\beta_{\rm pa5}"
latex_par_name["beta_pa6"] = r"\beta_{\rm pa6}"
latex_par_name["beta_ACT"] = r"\beta_{\rm ACT}"
latex_par_name["beta_ACT+komatsu"] = r"\beta_{\rm ACT + Komatsu}"
latex_par_name["beta_komatsu"] = r"\beta_{\rm Komatsu}"

cut_list = ["post_unblinding", "pre_unblinding"]
cut_list_name = ["baseline", "extended"]
folder_list = ["dr6", "dr6_weight"]
folder_list_name = ["uniform", "optimal"]

angle= {}
for folder, folder_name in zip(folder_list, folder_list_name):
    for cut, cut_name in zip(cut_list, cut_list_name):
        with open(f"{folder}/plots/results_EB_paper/angle_{cut}.pkl", "rb") as fp:
            angle[folder_name, cut_name] = pickle.load(fp)


alpha_list = ["alpha_pa5_f090", "alpha_pa5_f150", "alpha_pa6_f090", "alpha_pa6_f150"]


plt.figure(figsize=(20, 12))
plt.subplot(1,2,1)
for count, alpha in enumerate(alpha_list):
    color_list = ["red", "blue", "orange", "lightblue"]
    count_all = 0
    
    for shift1, folder_name in enumerate(folder_list_name):
        for shift2, cut_name in enumerate(cut_list_name):

    
            print(f"{alpha} {cut_name} ({folder_name} weighting)", angle[folder_name, cut_name][alpha, "mean"], angle[folder_name, cut_name][alpha, "std"])
            sh1 = shift1 * 0.1
            sh2 = shift2 * 0.2
            plt.errorbar(count + sh1 + sh2,
                         angle[folder_name, cut_name][alpha, "mean"],
                         angle[folder_name, cut_name][alpha, "std"],
                         label=f"{cut_name} ({folder_name} weight)",
                         color=color_list[count_all],
                         fmt="o")
            
            count_all += 1
    if count == 0:
        plt.legend(fontsize=26, loc="upper right")
        
xticks_list = ["pa5 f090", "pa5 f150", "pa6 f090", "pa6 f150"]
plt.xticks([0.15, 1.15, 2.15, 3.15], xticks_list, rotation=90, fontsize=32)
plt.ylabel(r"$\psi \ [\rm deg]$", fontsize=42)
plt.ylim(0,0.5)

beta_list = ["beta_pa5", "beta_pa6", "beta_ACT"]
plt.subplot(1,2,2)

for count, beta in enumerate(beta_list):
    count_all = 0
    for shift1, folder_name in enumerate(folder_list_name):
        for shift2, cut_name in enumerate(cut_list_name):

            print(f"{beta} {cut_name} ({folder_name} weighting)", angle[folder_name, cut_name][beta, "mean"], angle[folder_name, cut_name][beta, "std"]  )

            sh1 = shift1 * 0.1
            sh2 = shift2 * 0.2
            plt.errorbar(count + sh1 + sh2,
                        angle[folder_name, cut_name][beta, "mean"],
                        angle[folder_name, cut_name][beta, "std"],
                        label=f"{cut_name} ({folder_name} weighting)",
                        color=color_list[count_all],
                        fmt="o")
                        
            count_all += 1

plt.ylim(0,0.5)

xticks_list = ["pa5", "pa6", "ACT"]#, "ACT+Planck"]
plt.xticks([0.15,1.15,2.15], xticks_list, rotation=90, fontsize=32)
plt.ylabel(r"$\hat{\psi} \ [\rm deg]$", fontsize=42)

plt.tight_layout()
plt.savefig("all_alpha_and_beta.pdf")
plt.clf()
plt.close()

