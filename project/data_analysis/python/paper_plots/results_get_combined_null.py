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
plot_dir = f"plots/combined_null_test/"

pspy_utils.create_directory(plot_dir)

binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
iStart = 0
iStop =  1639

freq_pairs = ["90x90", "90x150", "150x150"]

ylim = {}
ylim["TT"] = (-10,10)
ylim["TE"] = (-3,3)
ylim["TB"] = (-3,3)
ylim["EE"] = (-2,2)
ylim["EB"] = (-1,1)
ylim["BB"] = (-1,1)

#plt.figure(figsize=(18,20))

combined_spectra = ["TT", "TE", "TB", "EE", "EB", "BB"]
for ispec, spectrum in enumerate(combined_spectra):
    
    plt.figure(figsize=(18,6))

    count=0

    for fp1 in freq_pairs:
        for fp2 in freq_pairs:
            if fp1 <= fp2: continue
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
            ndof = len(diff)
            pte = 1 - ss.chi2(ndof).cdf(chi2)

            std = np.sqrt(cov.diagonal())
            
            fa1, fb1 = fp1.split("x")
            fa2, fb2 = fp2.split("x")

            plt.errorbar(l1 - 8 + 8*count, diff, std, fmt="o", label=f"{fa1} GHz x {fb1} GHz - {fa2} GHz x {fb2} GHz, PTE: {pte*100:0.2f} %")
            count += 1
            
    lth = np.linspace(0, 8500)
    plt.plot(lth, lth*0, color="black", linestyle="--", alpha=0.5)
    plt.ylim(ylim[spectrum])
    plt.xlim(800, 4000)
    plt.legend(fontsize=16)
    plt.xlabel(r"$\ell$", fontsize=30)
    plt.ylabel(r"$D^{%s}_{\ell}$" % spectrum, fontsize=30)

    plt.savefig(f"{plot_dir}/null_{spectrum}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
