import pylab as plt
import numpy as np
import sys

from pspy import pspy_utils, so_dict, so_mcm, so_spectra
from pspipe_utils import log, pspipe_list



d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spec_dir = "spectra"
spec_corr_dir = "spectra_corrected"
plot_dir = "plots"
pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


cosmo_params = d["cosmo_params"]
l_th, ps_th = pspy_utils.ps_from_params(cosmo_params, type, 6000)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")


for spec in ["TB", "EB", "BB"]:

    plt.figure(figsize=(8,6))
    for spec_name in spec_name_list:
    
        name = spec_name.replace("spt_", "")
        fa, fb = name.split("x")
        if fa == fb:
            linestyle = "-"
        else:
            linestyle= "--"

        if spec in ["BB"]:
            plt.semilogy()
            plt.plot(l_th[:3500], ps_th[spec][:3500], color="gray")

    
        lb, Db_redo = so_spectra.read_ps(f"{spec_dir}/Dl_{spec_name}_noise.dat", spectra=spectra)
        id = np.where(lb>350)

        plt.plot(lb[id], Db_redo[spec][id], label=f"{spec} {name} (uncorrected)", linestyle=linestyle)
    plt.xlabel(r"$\ell$", fontsize=16)
    plt.ylabel(r"$N^{%s}_\ell$" % spec, fontsize=16)
    plt.legend()
    plt.savefig(f"{plot_dir}/noise_{spec}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
        

for spec in ["TT", "TE", "EE"]:
    plt.figure(figsize=(8,6))

    for spec_name in spec_name_list:
    
        name = spec_name.replace("spt_", "")
        fa, fb = name.split("x")
        if fa == fb:
            linestyle = "-"
        else:
            linestyle= "--"
            
        if spec in ["TT", "EE"]:
            plt.semilogy()
            plt.plot(l_th[:3500], ps_th[spec][:3500], color="gray")

        lb, Db_redo = so_spectra.read_ps(f"{spec_corr_dir}/Dl_{spec_name}_noise_tf_corr.dat", spectra=spectra)

        plt.plot(lb[id], Db_redo[spec][id], label=f"{spec} {name}", linestyle=linestyle)
    plt.xlabel(r"$\ell$", fontsize=16)
    plt.ylabel(r"$N^{%s}_\ell$" % spec, fontsize=16)
    plt.legend()
    plt.savefig(f"{plot_dir}/noise_{spec}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
