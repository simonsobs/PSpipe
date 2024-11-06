"""
This script look at the noise correlation between wafers
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, log, best_fits, external_data, misc
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

noise_dir = "results_noise"
pspy_utils.create_directory(noise_dir)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")


for spec in ["TT", "TE", "TB", "EE", "EB", "BB"]:

    plt.figure(figsize=(12, 8))

    for spec_name in spec_name_list:
        na, nb = spec_name.split("x")
        if na == nb: continue
        lb, Nb_ab = so_spectra.read_ps(f"spectra/Dl_{spec_name}_noise.dat", spectra=spectra)
        lb, Nb_aa = so_spectra.read_ps(f"spectra/Dl_{na}x{na}_noise.dat", spectra=spectra)
        lb, Nb_bb = so_spectra.read_ps(f"spectra/Dl_{nb}x{nb}_noise.dat", spectra=spectra)
        Rb = Nb_ab[spec] / np.sqrt(Nb_aa[spec] * Nb_bb[spec])
                
                
        spec_name = spec_name.replace("dr6_", "")
        spec_name = spec_name.replace("_", " ")
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.ylabel(r"$R^{\rm noise, %s}_\ell$" % spec, fontsize=22)

        plt.plot(lb, Rb, label=spec_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{noise_dir}/DR6_noise_correlation_{spec}.png", dpi=300)
    plt.clf()
    plt.close
