"""
This script compute combined null test
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import log
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import scipy.stats as ss
from matplotlib import rcParams
import matplotlib.ticker as ticker


labelsize = 14
fontsize = 20

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
ylim["TE"] =  (-1.2, 0.55)
ylim["TB"] = (-10,10)
ylim["EE"] = (.2, 4.3)
ylim["EB"] = (-1.2,1.2)
ylim["BB"] = (-1.2,1.2)

ylim_res = {}
ylim_res["TT"] = (-10,10)
ylim_res["TE"] = (-3,9)
ylim_res["TB"] = (-3,3)
ylim_res["EE"] = (-1.5,5)
ylim_res["EB"] = (-1,1)
ylim_res["BB"] = (-1,1)

#plt.figure(figsize=(18,20))
color_list = ["green", "red",  "blue"]
color_list_null = ["blue", "steelblue", "purple"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
combined_spectra = ["EE", "TE"]

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

shift_dict = {}
shift_dict["150x150"] = 0
shift_dict["90x150"] = -8
shift_dict["90x90"] = 8

divider_power = {}
divider_power["EE"] = 4
divider_power["TE"] = 8

f, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, height_ratios=[1, .75, 1, .75], dpi=100)

for ispec, spectrum in enumerate(combined_spectra):
    a0, a1 = axes[ispec*2:(ispec+1)*2]
    divider = 10 ** divider_power[spectrum]

    count=0
    
    a0.plot(lth, Dlth[spectrum] * lth ** lscale[spectrum] / divider, color="gray", linestyle="--", alpha=0.6)
    for my_c, fp in enumerate(freq_pairs[:]):
        fa, fb = fp.split("x")

        l, Dl_fg_sub, error = np.loadtxt(f"{combined_spec_dir}/Dl_{fp}_{spectrum}_cmb_only.dat", unpack=True)
        a0.errorbar(l + shift_dict[fp], Dl_fg_sub * l ** lscale[spectrum] / divider, error * l ** lscale[spectrum] / divider, fmt="o", label=f"{fa} GHz x {fb} GHz", color=color_list[my_c],
                    markersize=3, elinewidth=1, mfc='w')
    a0.legend(fontsize=labelsize)
    a0.set_xlim(500, 3000)
    a0.set_ylim(ylim[spectrum])
    a0.tick_params(axis='x', direction='in', labelbottom=False)

    if divider_power[spectrum] == 0:
        divider_str = ""
    else:
        divider_str = r"10^{%s}" % divider_power[spectrum]

    if lscale[spectrum] == 0:
        a0.set_ylabel(r"$ D^{%s}_{\ell} \ [{%s} \mu \rm K^{2}]$" % (spectrum, divider_str), fontsize=fontsize)
    elif lscale[spectrum] == 1:
        a0.set_ylabel(r"$ \ell D^{%s}_{\ell} \ [{%s} \mu \rm K^{2}]$" % (spectrum, divider_str), fontsize=fontsize)
    else:
        a0.set_ylabel(r"$ \ell^{%s} D^{%s}_{\ell} \ [{%s} \mu \rm K^{2}]$" % (lscale[spectrum], spectrum, divider_str), fontsize=fontsize)
    a0.tick_params(labelsize=labelsize)

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

            a1.errorbar(l1 - 8 + 8*count, diff, std, fmt="o", label=f"{fa1} GHz x {fb1} GHz - {fa2} GHz x {fb2} GHz, PTE: {pte*100:0.0f} %",
                        color=color_list_null[count], markersize=3, elinewidth=1, mfc='w')
            count += 1
            
    a1.plot(lth, lth*0, color="black", linestyle="--", alpha=0.5)
    a1.set_ylim(ylim_res[spectrum])
    a1.set_xlim(500, 3000)
    a1.legend(fontsize=labelsize)
    a1.set_xlabel(r"$\ell$", fontsize=fontsize)
    a1.set_ylabel(r"$\Delta D^{%s}_{\ell} \ [\mu \rm K^{2}]$" % spectrum, fontsize=fontsize)
    a1.tick_params(labelsize=labelsize)

    if ispec == 0:
        a1.tick_params(axis='x', direction='in', labelbottom=False)

plt.subplots_adjust(wspace=0, hspace=0)
f.align_ylabels()
# plt.show()
plt.savefig(f"{paper_plot_dir}/null_TEEE{tag}.pdf", bbox_inches="tight")
plt.clf()
plt.close()
