"""
This script performs null tests
and plot residual power spectra and a summary PTE distribution
"""

import numpy as np
import pylab as plt
import pickle
from pspy import pspy_utils
from matplotlib import rcParams


rcParams["font.family"] = "serif"
rcParams["font.size"] = "40"
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 16
rcParams["axes.titlesize"] = 16


def pte_histo(pte_list, n_bins, name, color):
    n_samples = len(pte_list)
    bins = np.linspace(0, 1, n_bins + 1)
    min, max = np.min(pte_list), np.max(pte_list)

    id_high = np.where(pte_list > 0.99)
    id_low = np.where(pte_list < 0.01)
    nPTE_high = len(id_high[0])
    nPTE_low =  len(id_low[0])
    
    plt.figure(figsize=(8,6))
    plt.title("Detectors test: In vs Out", fontsize=16)
    plt.xlabel(r"Probability to exceed (PTE)", fontsize=16)
    plt.hist(pte_list, bins=bins, label=f"n tests: {n_samples}, min: {min:.3f}, max: {max:.3f}", histtype='bar', facecolor=color, edgecolor='black', linewidth=3)
    plt.axhline(n_samples/n_bins, color="k", ls="--", alpha=0.5)
    plt.xlim(0,1)
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.savefig(f"{summary_dir}/pte_inout_{name}.png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    
test = "inout"
summary_dir = f"summary_{test}"
pspy_utils.create_directory(f"{summary_dir}")


label = "spectra_corrected+mc_cov+beam_cov+leakage_cov"
list = []
arrays = ["pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]

all_spec_list = [["TT", "TE", "ET", "EE"], ["EE", "EB", "BE", "BB"], ["TT", "TE", "ET", "TB", "BT", "EE", "EB", "BE", "BB"]]
file_name_list = ["T_and_E", "E_and_B", "all"]
colors = ["lightblue", "green", "orange"]



for color, name, spec_list in zip(colors, file_name_list, all_spec_list):
    list = []

    for ar in arrays:
        file = open(f"{ar}/plots/array_nulls/pte_dict.pkl", "rb")
        pte = pickle.load(file)
        for spec in spec_list:
            if (ar == "pa4_f220") & (spec != "TT"):
                continue
            print(ar, spec)
            list = np.append(list, pte[label, spec])
   # pte_histo(pte[label, "all"], n_bins)

    n_bins = 13
    print(len(list))
    print(np.min(list), np.max(list))
    pte_histo(list, n_bins, name, color)
