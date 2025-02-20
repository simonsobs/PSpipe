"""
This script compare the noise spectra with the signal power spectra
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, log, best_fits, external_data
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from matplotlib import rcParams

labelsize = 14
fontsize = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

remove_pa4_pol = True
nspec = len(spec_name_list)


paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

l_th, ps_th = so_spectra.read_ps("best_fits/cmb.dat", spectra=spectra)

yrange = {}
yrange["TT"] = (10, 2 * 10 ** 4)
yrange["TE"] = (-10, 10)
yrange["TB"] = (-10, 10)
yrange["EB"] = (-3, 3)

yrange["EE"] = (1, 500)

plt.figure(figsize=(12, 6), dpi=100)


my_colors = {}
my_colors["dr6_pa4_f220xdr6_pa4_f220"] = "gold"
my_colors["dr6_pa5_f090xdr6_pa5_f090"] = "red"
my_colors["dr6_pa5_f090xdr6_pa5_f150"] = "brown"
my_colors["dr6_pa5_f150xdr6_pa5_f150"] = "green"
my_colors["dr6_pa6_f090xdr6_pa6_f090"] = "orange"
my_colors["dr6_pa6_f090xdr6_pa6_f150"] = "deepskyblue"
my_colors["dr6_pa6_f150xdr6_pa6_f150"] = "blue"

["blue", "orange", "green", "red", "purple", "brown"]
count = 1
for spec in ["TT","EE"]:
    ax = plt.subplot(1,2, count)

    plt.semilogy()
    if count == 1:
        plt.plot(l_th, ps_th[spec], color="gray", label=r"$D^{\rm th}_{\ell}$")
    else:
        plt.plot(l_th, ps_th[spec], color="gray")
    
    # first construct inv_Nb_mean so it can be plotted 2nd rather than last
    inv_Nb_list = []
    for spec_name in spec_name_list:
        na, nb = spec_name.split("x")
        if ("pa4" in spec_name) & (spec != "TT") & remove_pa4_pol: continue
        sv_a, pa_a, ftag_a = na.split("_")
        sv_b, pa_b, ftag_b = nb.split("_")
        if pa_a != pa_b:   continue

        lb, Nb = so_spectra.read_ps(f"spectra/Dl_{spec_name}_noise.dat", spectra=spectra)

        if na == nb:
            inv_Nb_list += [ 1 / Nb[spec]]

    # plot inv_Nb_mean
    inv_Nb_mean = np.sum(inv_Nb_list, axis=0)
    if spec in ["TT", "EE", "BB"]:
        if na == nb:
            linestyle="-"
        else:
            linestyle="--"

        id = np.where(lb > 300)
        if count == 1:
            plt.plot(lb[id], 1 / inv_Nb_mean[id], linestyle=linestyle, label= f"DR6 effective noise", color="black")
        else:
            plt.plot(lb[id], 1 / inv_Nb_mean[id], linestyle=linestyle, color="black")

    # finally plot individual spectra
    for spec_name in spec_name_list:
        na, nb = spec_name.split("x")
        if ("pa4" in spec_name) & (spec != "TT") & remove_pa4_pol: continue
        sv_a, pa_a, ftag_a = na.split("_")
        sv_b, pa_b, ftag_b = nb.split("_")
        if pa_a != pa_b:   continue

        lb, Nb = so_spectra.read_ps(f"spectra/Dl_{spec_name}_noise.dat", spectra=spectra)

        if na == nb:
            linestyle="-"
        else:
            linestyle="--"
        
        id = np.where(lb > 300)
        if count == 1:
            plt.plot(lb[id], Nb[spec][id], linestyle=linestyle, label= f"{pa_a.upper()} ({ftag_a} x {ftag_b})", color=my_colors[spec_name])
        else:
            plt.plot(lb[id], Nb[spec][id], linestyle=linestyle, color=my_colors[spec_name])
        plt.xlabel(r"$\ell$", fontsize=fontsize)
        
        if count == 1:
            plt.ylabel(r"$\frac{\ell (\ell + 1)}{2\pi}N_\ell \ [\mu \rm K^{2}]$", fontsize=fontsize)

    plt.ylim(yrange[spec])
    plt.tick_params(labelsize=labelsize)
    plt.xlim(0, 4000)
    
    if spec == "TT":
        plt.title("TT Noise", fontsize=fontsize)
    else:
        plt.title("EE Noise", fontsize=fontsize)

    count += 1

plt.gcf().legend(fontsize=labelsize, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
    
plt.savefig(f"{paper_plot_dir}/DR6_noise.pdf", bbox_inches="tight")
plt.clf()
plt.close
