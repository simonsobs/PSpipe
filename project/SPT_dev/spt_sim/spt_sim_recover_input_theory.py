"""
This script check if we can recover the input theory using our power spectrum pipeline and SPT public simulations
"""

from pspy import pspy_utils, so_dict, so_mcm, so_spectra
from pspipe_utils import pspipe_list, log
import numpy as np
import pylab as plt
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

survey = "spt"
lmax = d["lmax"]
release_dir = d["release_dir"]

mcm_dir = "mcms"
sim_spec_dir = "sim_spectra_for_tf"
plot_dir = "plots_sim"
pspy_utils.create_directory(plot_dir)


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

input_spec_dir = f"{release_dir}/simulated_maps/input_maps/"

ps_in_list = {}
for iii in range(d["iStart"], d["iStop"]+1):
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")
        
    for spec_name in spec_name_list:
        lb, ps_in = so_spectra.read_ps(sim_spec_dir + f"/Dl_{spec_name}_nofilter_{iii:05d}.dat", spectra=spectra)
        for spec in spectra:
            if iii == 0: ps_in_list[spec_name, spec] = []
            ps_in_list[spec_name, spec] += [ps_in[spec]]

for spec_name in spec_name_list:

    _, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{spec_name}", spin_pairs=spin_pairs)

    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")
    
    # form the Dl used to generate simulation and bin it with Bbl
    input_Dl = {}
    for spec in spectra:
        if spec in ["TT", "TE", "EE", "BB"]:
            l, input_theory = np.loadtxt(f"{input_spec_dir}/input_cl_cmb_{spec.lower()}.txt", unpack=True)
            
            if spec != "TE":
                l, input_fg = np.loadtxt(f"{input_spec_dir}/input_cl_foregrounds_{fa}ghz{fb}ghz_{spec.lower()}.txt", unpack=True)
                input_Dl[spec] = (input_theory + input_fg) * l * (l + 1) / (2 * np.pi)
            else:
                input_Dl[spec] = input_theory * l * (l + 1) / (2 * np.pi)
                
            input_Dl[spec] = input_Dl[spec][2:lmax+2]
        else:
            l = np.arange(lmax)
            input_Dl[spec] = np.zeros(lmax)

    input_Dl["ET"] = input_Dl["TE"]
    input_Db = so_mcm.apply_Bbl(Bbl, input_Dl, spectra=spectra)
    
    for spec in spectra:
        mean = np.mean(ps_in_list[spec_name, spec], axis=0)
        std = np.std(ps_in_list[spec_name, spec], axis=0)
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.plot(lb, input_Db[spec])
        plt.errorbar(lb, mean, std, fmt=".")
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$D_\ell$", fontsize=14)
        plt.subplot(2,1,2)
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$D_\ell / D^{\rm redo}_\ell$", fontsize=14)
        plt.plot(lb, lb*0+1)
        plt.plot(lb, input_Db[spec]/mean, label=spec_name)
        plt.ylim(0.98, 1.02)
        plt.legend()
        plt.savefig(f"{plot_dir}/in_vs_th_{spec_name}_{spec}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
