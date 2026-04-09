"""
This script computes the filtering bias, an additive correction estimated by comparing masking_yes and masking_no spectra.
We compute it both with and without the alm mask, and compare it to SPT3G official correction.
The description of simulations is provided in section 3 of https://pole.uchicago.edu/public/data/quan26/index.html.
"""

from pspy import pspy_utils, so_dict, so_spectra
from pspipe_utils import pspipe_list, log
import numpy as np
import pylab as plt
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

survey = "spt"
lmax = d["lmax"]
type = d["type"]
release_dir = d["release_dir"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")


sim_spec_dir = "sim_spectra_for_tf"
plot_dir = "plots_sim"
tf_dir = "transfer_functions"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(tf_dir)

diff_list = {}

for iii in range(d["iStart"], d["iStop"]+1):
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")
        
    for spec_name in spec_name_list:
        lb, ps_filter_masking_yes = so_spectra.read_ps(sim_spec_dir + f"/Dl_{spec_name}_filter_masking_yes_{iii:05d}.dat", spectra=spectra)
        lb, ps_filter_masking_no = so_spectra.read_ps(sim_spec_dir + f"/Dl_{spec_name}_filter_masking_no_{iii:05d}.dat", spectra=spectra)

        lb, ps_filter_masking_yes_alm_mask = so_spectra.read_ps(sim_spec_dir + f"/Dl_{spec_name}_filter_masking_yes_alm_mask_{iii:05d}.dat", spectra=spectra)
        lb, ps_filter_masking_no_alm_mask = so_spectra.read_ps(sim_spec_dir + f"/Dl_{spec_name}_filter_masking_no_alm_mask_{iii:05d}.dat", spectra=spectra)

        for spec in spectra:
            if iii == 0:
                diff_list["no_alm_mask", spec_name, spec] = []
                diff_list["alm_mask", spec_name, spec] = []

            diff_list["no_alm_mask", spec_name, spec] += [ps_filter_masking_yes[spec] - ps_filter_masking_no[spec]]
            diff_list["alm_mask", spec_name, spec] += [ps_filter_masking_yes_alm_mask[spec] - ps_filter_masking_no_alm_mask[spec]]


for spec_name in spec_name_list:

    additive_bias_no_alm_mask = {}
    additive_bias_alm_mask = {}
    
    for spec in spectra:
        additive_bias_no_alm_mask[spec] = np.mean(diff_list["no_alm_mask", spec_name, spec], axis=0)
        additive_bias_alm_mask[spec] = np.mean(diff_list["alm_mask", spec_name, spec], axis=0)
    
    so_spectra.write_ps(tf_dir + f"/additive_bias_no_alm_mask_{spec_name}.dat", lb, additive_bias_no_alm_mask, type, spectra=spectra)
    so_spectra.write_ps(tf_dir + f"/additive_bias_alm_mask_{spec_name}.dat", lb, additive_bias_alm_mask, type, spectra=spectra)


    # no compare with spt
    spt_tf_dir = f"{release_dir}/ancillary_products/specific_to_c25/"
    
    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")
    
    for mode in ["tt", "te", "et", "ee"]:
        filter_artefact_bias_file = f"{spt_tf_dir}/filtering_artifact_bias_{fa}ghz{fb}ghz_{mode}.txt"
        l,  filter_artefact_bias = np.loadtxt(filter_artefact_bias_file, unpack=True)
        filter_artefact_bias *= l * (l + 1) / (2 * np.pi) #corrections are in Cl

        plt.plot(l, filter_artefact_bias, label= "camphuis bias")
        plt.plot(lb, additive_bias_no_alm_mask[mode.upper()], label= "additive bias (no alm mask)")
        plt.plot(lb, additive_bias_alm_mask[mode.upper()], label= "additive bias (alm mask)")
        plt.legend()
        plt.show()
