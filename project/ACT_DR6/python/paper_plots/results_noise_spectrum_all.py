"""
This script compare the noise spectra with the signal power spectra
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, log, best_fits, external_data
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from matplotlib import rcParams


rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

remove_pa4_pol = True
nspec = len(spec_name_list)


noise_dir = "results_noise"

pspy_utils.create_directory(noise_dir)


l_th, ps_th = so_spectra.read_ps("best_fits/cmb.dat", spectra=spectra)

yrange = {}
yrange["TT"] = (10, 2 * 10 ** 4)
yrange["TE"] = (-10, 10)
yrange["TB"] = (-10, 10)
yrange["EB"] = (-3, 3)

yrange["EE"] = (1, 500)

plt.figure(figsize=(14, 8))


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
    plt.plot(l_th, ps_th[spec], color="gray", label=r"$D^{\rm th}_{\ell}$")
        
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
            linestyle="-"
        else:
            linestyle="--"


        id = np.where(lb > 300)
        plt.plot(lb[id], Nb[spec][id], linestyle=linestyle, label= f"{pa_a} {ftag_a} x {pa_b} {ftag_b}", color=my_colors[spec_name])
        plt.xlabel(r"$\ell$", fontsize=22)
        
        if count == 1:
            plt.ylabel(r"$\frac{\ell (\ell + 1)}{2\pi}N_\ell \ [\mu K^{2}]$", fontsize=22)

    inv_Nb_mean = np.sum(inv_Nb_list, axis=0)
    if spec in ["TT", "EE", "BB"]:
        plt.plot(lb[id], 1 / inv_Nb_mean[id], linestyle=linestyle, label= f"DR6 effective noise", color="black")


    plt.ylim(yrange[spec])
    plt.xlim(0, 4000)
    
    if spec == "TT":
        plt.title("Temperature noise")
    else:
        plt.title("E-modes noise")

    if count>1:
        ax.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    
    count += 1
    
plt.tight_layout()
plt.savefig(f"{noise_dir}/DR6_noise.png", dpi=300)
plt.clf()
plt.close
