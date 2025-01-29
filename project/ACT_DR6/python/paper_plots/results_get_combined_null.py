"""
This script compute combined null test
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import log
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from matplotlib import rcParams
import matplotlib.ticker as ticker


rcParams["font.family"] = "serif"
rcParams["font.size"] = "20"
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]
combined_spec_dir = f"combined_spectra{tag}"

combined_sim_spec_dir = f"combined_sim_spectra_syst"
bestfit_dir = f"best_fits{tag}"

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)


binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
iStart = 0
iStop = 900
print(iStart, iStop)
freq_pairs = ["90x90", "90x150", "150x150"]

lscale = {}
lscale["TT"] = 2
lscale["TE"] = 2
lscale["TB"] = 0
lscale["EE"] = 1
lscale["EB"] = 0
lscale["BB"] = 0


ylim= {}
ylim["TT"] = (-1.5*10**7, 1.8*10**9)
ylim["TE"] =  (-1.2*10**8, 0.55*10**8)
ylim["TB"] = (-10,10)
ylim["EE"] = (2*10**3, 4.3*10**4)
ylim["EB"] = (-1.2,1.2)
ylim["BB"] = (-1.2,1.2)

ylim_res = {}
ylim_res["TT"] = (-10,10)
ylim_res["TE"] = (-3,5)
ylim_res["TB"] = (-3,3)
ylim_res["EE"] = (-2,3.5)
ylim_res["EB"] = (-1,1)
ylim_res["BB"] = (-1,1)

#plt.figure(figsize=(18,20))
color_list = ["green", "red",  "blue"]
color_list_null = ["blue", "steelblue", "purple"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
combined_spectra = ["TT", "TE", "TB", "EE", "EB", "BB"]
combined_spectra = ["TE", "EE"]

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

shift_dict = {}
shift_dict["150x150"] = 0
shift_dict["90x150"] = -8
shift_dict["90x90"] = 8



for ispec, spectrum in enumerate(combined_spectra):
    f, (a0, a1) = plt.subplots(2, 1, height_ratios=[1.5, 1], figsize=(20, 10))


    count=0
    
    a0.plot(lth, Dlth[spectrum] * lth ** lscale[spectrum], color="gray", linestyle="--", alpha=0.6)
    for my_c, fp in enumerate(freq_pairs[:]):
        fa, fb = fp.split("x")

        l, Dl_fg_sub, error = np.loadtxt(f"{combined_spec_dir}/Dl_{fp}_{spectrum}_cmb_only.dat", unpack=True)
        a0.errorbar(l + shift_dict[fp], Dl_fg_sub * l ** lscale[spectrum], error * l ** lscale[spectrum], fmt=".", label=f"{fa} GHz x {fb} GHz", color=color_list[my_c], mfc='w', markersize=8)
    a0.legend(fontsize=16)
    a0.set_xlim(500, 3000)
    a0.set_ylim(ylim[spectrum])
    a0.set_xticks([])
    

    if lscale[spectrum] == 0:
        a0.set_ylabel(r"$ D^{%s}_{\ell} \ [\mu K^{2}]$" % spectrum, fontsize=30)
    elif lscale[spectrum] == 1:
        a0.set_ylabel(r"$ \ell D^{%s}_{\ell} \ [\mu K^{2}]$" % (spectrum), fontsize=30)
    else:
        a0.set_ylabel(r"$ \ell^{%s} D^{%s}_{\ell} \ [\mu K^{2}]$" % (lscale[spectrum], spectrum), fontsize=30)

    for fp1 in freq_pairs:
        for fp2 in freq_pairs:
            if fp1 <= fp2: continue
            print(spectrum, fp1,fp2)
            l1, Dl1_fg_sub, _ = np.loadtxt(f"{combined_spec_dir}/Dl_{fp1}_{spectrum}_cmb_only.dat", unpack=True)
            l2, Dl2_fg_sub, _ = np.loadtxt(f"{combined_spec_dir}/Dl_{fp2}_{spectrum}_cmb_only.dat", unpack=True)
            
            min_l_null = np.maximum(l1[0], l2[0])
            id1 = np.where(l1 >= min_l_null)
            id2 = np.where(l2 >= min_l_null)
            l1 = l1[id1]
            l2 = l2[id2]
            diff = (Dl1_fg_sub[id1] - Dl2_fg_sub[id2])
            
            mc_diff_list = []
            for iii in range(iStart, iStop + 1):
                l1_, Dl1_fg_sub_sim, _ = np.loadtxt(f"{combined_sim_spec_dir}/Dl_{fp1}_{spectrum}_{iii:05d}_cmb_only.dat", unpack=True)
                l2_, Dl2_fg_sub_sim, _ = np.loadtxt(f"{combined_sim_spec_dir}/Dl_{fp2}_{spectrum}_{iii:05d}_cmb_only.dat", unpack=True)
                diff_sim = (Dl1_fg_sub_sim[id1] - Dl2_fg_sub_sim[id2])
                mc_diff_list  += [diff_sim]

            cov = np.cov(mc_diff_list, rowvar=False)
            corr = so_cov.cov2corr(cov, remove_diag=True)
            
            chi2 = diff @ np.linalg.inv(cov) @ diff
            ndof = len(diff) - 1 # spectra are calibrated
            pte = 1 - ss.chi2(ndof).cdf(chi2)

            std = np.sqrt(cov.diagonal())
            
            fa1, fb1 = fp1.split("x")
            fa2, fb2 = fp2.split("x")

            a1.errorbar(l1 - 8 + 8*count, diff, std, fmt="o", label=f"{fa1} GHz x {fb1} GHz - {fa2} GHz x {fb2} GHz, PTE: {pte*100:0.0f} %", color=color_list_null[count], mfc='w')
            count += 1
            
    a1.plot(lth, lth*0, color="black", linestyle="--", alpha=0.5)
    a1.set_ylim(ylim_res[spectrum])
    a1.set_xlim(500, 3000)
    a1.legend(fontsize=16)
    a1.set_xlabel(r"$\ell$", fontsize=30)
    a1.set_ylabel(r"$\Delta D^{%s}_{\ell} \ [\mu K^{2}]$" % spectrum, fontsize=30)
    plt.subplots_adjust(wspace=0, hspace=0)
    f.align_ylabels()
   # plt.show()
    plt.savefig(f"{paper_plot_dir}/null_{spectrum}{tag}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()
