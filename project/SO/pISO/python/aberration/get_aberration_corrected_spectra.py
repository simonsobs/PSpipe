"""
This script correct the effect of aberration on the data power spectrum
"""

import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra
from pspipe_utils import pspipe_list, log
import numpy as np
import pylab as plt
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

type = d["type"]
surveys = d["surveys"]
lmax = d["lmax"]

ab_dir = d["aberration_correction"]

plot_dir = "plots/aberration_data"
spec_dir = "spectra_leak_corr"
spec_corr_dir = spec_dir + "_ab_corr"

pspy_utils.create_directory(spec_corr_dir)
pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

for spec_name in spec_name_list:

    log.info(f"correct {spec_name} ")
    
    lb, ab_corr = so_spectra.read_ps(f"{ab_dir}/aberration_correction_{spec_name}.dat", spectra=spectra)
    lb, ps = so_spectra.read_ps(spec_dir + f"/{type}_{spec_name}_cross.dat", spectra=spectra)
    
    ps_corr = {}
    for spectrum in spectra:

        ps_corr[spectrum] =  ps[spectrum] - ab_corr[spectrum] #ab_corr is defined as aberrated - non aberrated
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.errorbar(lb, ps[spectrum], fmt=".", color="red", label = "aberrated")
        plt.errorbar(lb, ps_corr[spectrum], fmt=".", color="blue", label = "corrected")
        plt.ylabel(r"$D^{%s}_{\ell}$" % spectrum , fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.legend(fontsize=18)
        plt.subplot(2,1,2)
        plt.errorbar(lb, ps[spectrum] - ps_corr[spectrum], color="blue")
        plt.ylabel(r"$D^{%s, aberrated}_{\ell}  - D^{%s}_{\ell}$" % (spectrum, spectrum), fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.savefig(f"{plot_dir}/{spec_name}_{spectrum}.png", bbox_inches="tight")
        plt.clf()
        plt.close()

    so_spectra.write_ps(spec_corr_dir + f"/{type}_{spec_name}_cross.dat",
                        lb,
                        ps_corr,
                        type=type,
                        spectra=spectra)
