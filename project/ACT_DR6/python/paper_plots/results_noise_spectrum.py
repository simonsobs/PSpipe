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

noise_dir = "plots/results_noise"
pspy_utils.create_directory(noise_dir)

l_th, ps_th = so_spectra.read_ps("best_fits/cmb.dat", spectra=spectra)

yrange = {}
yrange["TT"] = (5, 2 * 10 ** 4)
yrange["TE"] = (-10, 10)
yrange["TB"] = (-10, 10)
yrange["EB"] = (-3, 3)

yrange["EE"] = (0.1, 2 * 10 ** 3)

for spec in ["TT", "TE", "TB", "EE", "EB"]:
    plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)

    if spec == "TE":
        plt.plot(l_th, ps_th[spec] * 0.1, color="gray", label=r"1/10 $D^{%s}_{\ell}$" % spec)
    else:
        plt.plot(l_th, ps_th[spec], color="gray", label=r"$D^{%s}_{\ell}$" % spec)


    if spec in ["TT", "EE", "BB"]:
        plt.semilogy()
        
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
        plt.plot(lb[id], Nb[spec][id], linestyle=linestyle, label= f"{pa_a} {ftag_a} x {pa_b} {ftag_b}")
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.ylabel(r"$\frac{\ell (\ell + 1)}{2\pi}N^{%s}_\ell \ [\mu K^{2}]$" % spec, fontsize=22)

    inv_Nb_mean = np.sum(inv_Nb_list, axis=0)
    if spec in ["TT", "EE", "BB"]:
        plt.plot(lb[id], 1 / inv_Nb_mean[id], linestyle=linestyle, label= f"DR6 effective noise", color="black")

    np.savetxt(f"{noise_dir}/DR6_noise_combined_{spec}.dat", np.transpose([lb, 1 / inv_Nb_mean]))
    plt.ylim(yrange[spec])
    plt.xlim(0, 6000)
    ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{noise_dir}/DR6_noise_{spec}.pdf")
    plt.clf()
    plt.close
