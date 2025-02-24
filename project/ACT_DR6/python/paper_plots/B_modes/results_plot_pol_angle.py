"""
This script produce lot of different plots of the pol angles
"""

from pspy import so_dict, pspy_utils, so_spectra
from pspipe_utils import pol_angle
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from cobaya.run import run
from getdist import loadMCSamples, MCSamples
from matplotlib import rcParams
import pickle


rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

def gaussian(mean, std):
    x = np.linspace(-1, 1, 1000)
    gauss = ss.norm.pdf(x, mean, std)
    gauss /= np.max(gauss)
    return x, gauss
    
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

tag = d["best_fit_tag"]

binning_file = d["binning_file"]
lmax = d["lmax"]
bestfit_dir = f"best_fits{tag}"
result_dir = f"plots/results_EB{tag}"

roots = ["mcmc"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
params = ["alpha_pa5_f090", "alpha_pa5_f150", "alpha_pa6_f090", "alpha_pa6_f150"]

syst_error  = 0.1
cut = "post_unblinding"

print("*********")
print(cut)

burnin = 0.5
samples = loadMCSamples(f"{result_dir}/chains_{cut}/{roots[0]}", settings={"ignore_rows": burnin})
corr = samples.corr(params)
cov = samples.cov(params)
std =  samples.std(params)


latex_par_name = {}
latex_par_name["alpha_pa5_f090"] = r"\alpha_{pa5 f090}"
latex_par_name["alpha_pa6_f090"] = r"\alpha_{pa6 f090}"
latex_par_name["alpha_pa5_f150"] = r"\alpha_{pa5 f150}"
latex_par_name["alpha_pa6_f150"] = r"\alpha_{pa6 f150}"
latex_par_name["beta_pa5"] = r"\beta_{\rm pa5}"
latex_par_name["beta_pa6"] = r"\beta_{\rm pa6}"
latex_par_name["beta_ACT"] = r"\beta_{\rm ACT}"
latex_par_name["beta_ACT+komatsu"] = r"\beta_{\rm ACT + Komatsu}"
latex_par_name["beta_komatsu"] = r"\beta_{\rm Komatsu}"


mean, std = {}, {}
all_angles = {}

plt.figure(figsize=(14,8))
mean_ml_alpha = 0
cov_ml_alpha = 0
for par_name in params:
    mean[par_name] = samples.mean(par_name)
    std[par_name] = samples.std(par_name)
    print(par_name, mean[par_name], std[par_name])
    x, gauss = gaussian(mean[par_name], std[par_name] )
    plt.plot(x, gauss, label=f"${latex_par_name[par_name]}$ = {mean[par_name]:.3f} $\pm$ {std[par_name]:.3f}")
    
    cov_ml_alpha += 1 / std[par_name]**2
    mean_ml_alpha += mean[par_name] / std[par_name] ** 2

    all_angles[par_name, "mean"] = mean[par_name]
    all_angles[par_name, "std"] = std[par_name]

plt.xlim(-0.3, 0.8)
plt.xlabel(r"$\alpha$", fontsize=25)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(f"{paper_plot_dir}/alpha_ACT_{cut}{tag}.pdf")
plt.clf()
plt.close()
    
cov_ml_alpha = 1 / cov_ml_alpha
mean_ml_alpha = cov_ml_alpha * mean_ml_alpha
std_ml_alpha = np.sqrt(cov_ml_alpha)

all_angles["alpha_all", "mean"] = mean_ml_alpha
all_angles["alpha_all", "std"] = std_ml_alpha


cov_90 = 1 / (1 / std["alpha_pa5_f090"] ** 2 + 1 / std["alpha_pa6_f090"] ** 2)
mean_90 = cov_90 * (mean["alpha_pa5_f090"]  / std["alpha_pa5_f090"] ** 2 + mean["alpha_pa6_f090"]  / std["alpha_pa6_f090"] ** 2)
std_90 =  np.sqrt(cov_90)


cov_150 = 1 / (1 / std["alpha_pa5_f150"] ** 2 + 1 / std["alpha_pa6_f150"] ** 2)
mean_150 = cov_150 * (mean["alpha_pa5_f150"]  / std["alpha_pa5_f150"] ** 2 + mean["alpha_pa6_f150"]  / std["alpha_pa6_f150"] ** 2)
std_150 =  np.sqrt(cov_150)

cov_pa5 = 1 / (1 / std["alpha_pa5_f090"] ** 2 + 1 / std["alpha_pa5_f150"] ** 2)
mean_pa5 = cov_pa5 * (mean["alpha_pa5_f090"]  / std["alpha_pa5_f090"] ** 2 + mean["alpha_pa5_f150"]  / std["alpha_pa5_f150"] ** 2)
std_pa5 =  np.sqrt(cov_pa5)

cov_pa6 = 1 / (1 / std["alpha_pa6_f090"] ** 2 + 1 / std["alpha_pa6_f150"] ** 2)
mean_pa6 = cov_pa6 * (mean["alpha_pa6_f090"]  / std["alpha_pa6_f090"] ** 2 + mean["alpha_pa6_f150"]  / std["alpha_pa6_f150"] ** 2)
std_pa6 =  np.sqrt(cov_pa6)




all_angles["alpha_90", "mean"] = mean_90
all_angles["alpha_90", "std"] = std_90

all_angles["alpha_150", "mean"] = mean_150
all_angles["alpha_150", "std"] = std_150

print("alpha 150 GHz", mean_150, std_150)
print("alpha 90 GHz", mean_90, std_90)
print("alpha pa5 GHz", mean_pa5, std_pa5)
print("alpha pa6 GHz", mean_pa6, std_pa6)

print("alpha_all", mean_ml_alpha, std_ml_alpha)

    
for c1, par_name1 in enumerate(params):
    for c2, par_name2 in enumerate(params):
        if c1 >= c2:  continue
        diff = (mean[par_name1] - mean[par_name2])/np.sqrt(std[par_name1] ** 2 + std[par_name2] ** 2)
        print(f"{par_name1} - {par_name2}, {diff} sigma")


plt.figure(figsize=(14,8))
mean_comb, std_comb = {}, {}
for det_waf in ["pa5", "pa6"]:
    cov_comb = 1 / (1 / std[f"alpha_{det_waf}_f090"] ** 2 + 1 / std[f"alpha_{det_waf}_f150"] ** 2 )
    mean_comb[det_waf] = cov_comb * ( mean[f"alpha_{det_waf}_f090"] / std[f"alpha_{det_waf}_f090"] ** 2 + mean[f"alpha_{det_waf}_f150"] / std[f"alpha_{det_waf}_f150"] ** 2)
    std_comb[det_waf] =  np.sqrt(cov_comb)
    std_comb[det_waf] = np.sqrt(std_comb[det_waf] ** 2 + syst_error ** 2) # add sys error
    print("beta", det_waf, mean_comb[det_waf], std_comb[det_waf])


    x, gauss = gaussian(mean_comb[det_waf], std_comb[det_waf] )
    
    plt.plot(x, gauss, label=f"${latex_par_name[f'beta_{det_waf}']}$ = {mean_comb[det_waf]:.3f} $\pm$ {std_comb[det_waf]:.3f}", linestyle="--", alpha=0.5)
    
    all_angles[f"beta_{det_waf}", "mean"] = mean_comb[det_waf]
    all_angles[f"beta_{det_waf}", "std"] = std_comb[det_waf]



    
cov_ACT = 1 / ( 1 / std_comb["pa5"] ** 2 + 1 / std_comb["pa6"] ** 2 )
mean_ACT = cov_ACT * (mean_comb["pa5"] / std_comb["pa5"] ** 2 + mean_comb["pa6"] / std_comb["pa6"] ** 2)
std_ACT = np.sqrt(cov_ACT)

print("beta_all", mean_ACT, std_ACT)


x, gauss = gaussian(mean_ACT, std_ACT)
plt.plot(x, gauss, label=f"${latex_par_name[f'beta_ACT']}$ = {mean_ACT:.3f} $\pm$ {std_ACT:.3f}", color="gray")

mean_komatsu = 0.342 # https://arxiv.org/abs/2205.13962
std_komatsu = 0.094

cov_ACT_komatsu = 1 / ( 1 / std_ACT ** 2 + 1 / std_komatsu ** 2 )
mean_ACT_komatsu = cov_ACT_komatsu * (mean_ACT / std_ACT ** 2 + mean_komatsu / std_komatsu ** 2)
std_ACT_komatsu = np.sqrt(cov_ACT_komatsu)


x, gauss = gaussian(mean_komatsu, std_komatsu)
plt.plot(x, gauss, label=f"${latex_par_name[f'beta_komatsu']}$ = {mean_komatsu:.3f} $\pm$ {std_komatsu:.3f}")

x, gauss = gaussian(mean_ACT_komatsu, std_ACT_komatsu)
plt.plot(x, gauss, label=f"${latex_par_name[f'beta_ACT+komatsu']}$ = {mean_ACT_komatsu:.3f} $\pm$ {std_ACT_komatsu:.3f}")
plt.xlim(-0.3, 0.8)
plt.legend(fontsize=15)
plt.xlabel(r"$\beta$", fontsize=25)
plt.tight_layout()
plt.savefig(f"{paper_plot_dir}/beta_{cut}{tag}.pdf")
plt.clf()
plt.close()


all_angles[f"beta_ACT", "mean"] = mean_ACT
all_angles[f"beta_ACT", "std"] = std_ACT
all_angles[f"beta_komatsu", "mean"] = mean_komatsu
all_angles[f"beta_komatsu", "std"] = std_komatsu
all_angles[f"beta_ACT+komatsu", "mean"] = mean_ACT_komatsu
all_angles[f"beta_ACT+komatsu", "std"] = std_ACT_komatsu


with open(f"{result_dir}/angle_{cut}.pkl", "wb") as fp:
    pickle.dump(all_angles, fp)





def plot_spectrum(l, psth_rot, lb_dat, ps, error, cov, spectrum):


    lb, ps_th_b = pspy_utils.naive_binning(l, psth_rot[spectrum], binning_file, lmax)
    id = np.where(lb>=lb_dat[0])
    lb, ps_th_b = lb[id], ps_th_b[id]
    chi2 = (ps - ps_th_b) @ np.linalg.inv(cov) @ (ps - ps_th_b)
    chi2_null = (ps) @ np.linalg.inv(cov) @ (ps)

    print("delta chi2", chi2 - chi2_null)
    pte = 1 - ss.chi2(len(lb)).cdf(chi2)
    pte_null = 1 - ss.chi2(len(lb)).cdf(chi2_null)
    plt.figure(figsize=(14,8))
    plt.errorbar(l, psth_rot[spectrum] * 0, label=r"PTE ($\psi=0^{\circ}$):  %.5f" % (pte_null*100)  + "%", color="black", linestyle="--")
    plt.errorbar(l, psth_rot[spectrum], label=r"PTE ($\psi=%.2f^{\circ}$):  %.1f" % (mean_ACT, pte*100)  + "%", color="gray")
    plt.errorbar(lb_dat, ps, error, fmt="o", color="royalblue")
    plt.xlabel(r"$\ell$", fontsize=35)
    plt.ylabel(r"$D^{\rm %s}_{\ell} \ [\mu \rm K^{2}]$" % spectrum, fontsize=35)
    plt.legend(fontsize=20)
    if spectrum == "TB":
        plt.ylim(-4, 6)
    if spectrum == "EB":
        plt.ylim(-0.3, 0.6)

    plt.xlim(0,3500)
    plt.tight_layout()
    plt.savefig(f"{paper_plot_dir}/combined_{spectrum}_{cut}{tag}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()


print("*********")

l, ps_th = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)
l, psth_rot = pol_angle.rot_theory_spectrum(l, ps_th, mean_ACT, mean_ACT)
lb_dat, ps, error = np.loadtxt(f"{result_dir}/combined_EB_{cut}.dat", unpack=True)
cov = np.load(f"{result_dir}/combined_cov_EB_{cut}.npy")

plot_spectrum(l, psth_rot, lb_dat, ps, error, cov, "EB")

if cut == "post_unblinding": #For TB we have only the post_unblinding spectrum available
    combined_spec_dir = f"combined_spectra{tag}"

    lb_dat, ps, error = np.loadtxt(f"{combined_spec_dir}/Dl_all_TB_cmb_only.dat", unpack=True)
    cov = np.load(f"{combined_spec_dir}/cov_all_TB.npy")
    plot_spectrum(l, psth_rot, lb_dat, ps, error, cov, "TB")

