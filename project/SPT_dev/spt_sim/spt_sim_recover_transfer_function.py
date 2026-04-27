"""
This script retrieves the spt transfer function from the analysis of simulation.
"""

from pspy import pspy_utils, so_dict, so_mcm, so_spectra
from pspipe_utils import pspipe_list, log
import numpy as np
import pylab as plt
import healpy as hp
import sys
import time
from scipy.interpolate import interp1d


def get_Bbl_dict(Bbl):
    Bbl_dict = {}
    Bbl_dict["tt"] = Bbl["spin0xspin0"]
    Bbl_dict["te"] = Bbl["spin0xspin0"]
    Bbl_dict["et"] = Bbl["spin0xspin0"]
    nbins, my_lmax = int(Bbl["spin2xspin2"].shape[0]/4), int(Bbl["spin2xspin2"].shape[1]/4)
    Bbl_dict["ee"] = Bbl["spin2xspin2"][:nbins, :my_lmax]
    return Bbl_dict


def correct_spt_additive_bias(lb, ps, spec_name, Bbl):
    """
    correct spt additive inpainting and filtering bias, note that the bias are given in Dl
    """
    from copy import deepcopy

    ps_corrected = deepcopy(ps)
    
    tf_dir_camphuis_dir = f"{release_dir}/ancillary_products/specific_to_c25/"
    
    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")
    
    Bbl_dict = get_Bbl_dict(Bbl)

    additive_bias = {}
    for mode in ["tt", "te", "et", "ee"]:
        filter_artefact_bias_file = f"{tf_dir_camphuis_dir}/filtering_artifact_bias_{fa}ghz{fb}ghz_{mode}.txt"
        l,  filter_artefact_bias = np.loadtxt(filter_artefact_bias_file, unpack=True)
        additive_bias[mode] = (filter_artefact_bias) * l * (l + 1) / (2 * np.pi) #corrections are in Cl
        additive_bias[mode] =  np.dot(Bbl_dict[mode], additive_bias[mode][:lmax])
        ps_corrected[mode.upper()] = ps[mode.upper()] - additive_bias[mode]
        
    return lb, ps_corrected


def interpolate_tf(lb, mean_tf, ell_min_zero=400):
    
    ell = np.arange(2, 4001) # this is to follow spt convention
    f_interp = interp1d(lb[id], mean, kind='cubic', bounds_error=False, fill_value="extrapolate")
    Dell = f_interp(ell)
    
    Dell[:ell_min_zero] = 0
    ell_max_bin = int(lb[-1])
    Dell[ell_max_bin:] = Dell[ell_max_bin]
    
    return ell, Dell


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

survey = "spt"
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
binned_mcm = d["binned_mcm"]
release_dir = d["release_dir"]
camp_dir = f"{release_dir}/ancillary_products/specific_to_c25/"
also_masking_no = True


mcm_dir = "mcms"
plot_dir = "plots_sim"
sim_spec_dir = "sim_spectra_for_tf"
tf_dir = "transfer_functions"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(tf_dir)


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
arrays_spt = d["arrays_spt"]

tf = {}

l, pixwin_l = np.loadtxt(f"{release_dir}/ancillary_products/generally_applicable/pixel_window_function.txt", unpack=True)
lb, xtra_pw = pspy_utils.naive_binning(np.arange(len(pixwin_l)),  pixwin_l, binning_file, lmax)


cases = ["filter_masking_yes", "nofilter_alm_mask", "filter_masking_yes_alm_mask"]
if also_masking_no == True:
    cases += ["filter_masking_no", "filter_masking_no_alm_mask"]



for iii in range(d["iStart"], d["iStop"] + 1):
    print(iii)
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")
        
    n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
        
    for i_spec in range(n_spec):
        sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]
        spec_name = f"{sv1}_{ar1}x{sv2}_{ar2}"
        _, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{spec_name}", spin_pairs=spin_pairs)

        for spec in spectra:

            lb, ps_nofilter = so_spectra.read_ps(sim_spec_dir + f"/Dl_{spec_name}_nofilter_{iii:05d}.dat", spectra=spectra)
            id = np.where(lb>=400)

            for case in cases:
            
                if iii == 0: tf[case, spec_name, spec] = []
                lb, ps = so_spectra.read_ps(sim_spec_dir + f"/Dl_{spec_name}_{case}_{iii:05d}.dat", spectra=spectra)
                lb, ps = correct_spt_additive_bias(lb, ps, spec_name, Bbl)
                tf[case, spec_name, spec] += [ps[spec][id]/ps_nofilter[spec][id]]


for i_spec in range(n_spec):
    sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]
    spec_name = f"{sv1}_{ar1}x{sv2}_{ar2}"
    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")

    for case in cases:
        for spec in spectra:
        
    
            mean = np.mean(tf[case, spec_name, spec], axis=0)
            std = np.std(tf[case, spec_name, spec], axis=0)
            
            ell, Dell  = interpolate_tf(lb, mean, ell_min_zero=400)
            
            if spec[0] == spec[1]:
                plt.plot(ell, Dell)
                plt.errorbar(lb[id], mean, std, fmt=".", label=f"{spec_name} {spec} {case}")
                
                if (case in ["filter_masking_yes_alm_mask", "filter_masking_no_alm_mask"]) & (spec in ["TT", "EE"]):
                
                    tf_file = f"{camp_dir}/filter_transfer_function_c25_v1_{fa}ghz{fb}ghz_{spec.lower()}.txt"
                    l, tf_camphuis = np.loadtxt(tf_file, unpack=True)
                    plt.plot(l, tf_camphuis, label="camphuis")

                plt.legend()
                plt.savefig(f"{plot_dir}/transfer_function_{case}_{spec}_{spec_name}.png", bbox_inches="tight")
                plt.clf()
                plt.close()

                np.savetxt(f"{tf_dir}/transfer_function_{case}_{spec}_{spec_name}.dat", np.transpose([lb[id], mean, std]))

# overplot TT, EE, BB
for i_spec in range(n_spec):
    sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]
    spec_name = f"{sv1}_{ar1}x{sv2}_{ar2}"

    for case in cases:
    
        plt.figure(figsize=(12,8))
        for spec in ["TT", "EE", "BB"]:
            mean = np.mean(tf[case, spec_name, spec], axis=0)
            std = np.std(tf[case, spec_name, spec], axis=0)
            
            plt.errorbar(lb[id], mean, std, fmt=".", label=f"{spec_name} {spec} {case}")

        plt.legend()
        plt.savefig(f"{plot_dir}/all_spec_transfer_function_{case}_{spec_name}.png", bbox_inches="tight")
        plt.clf()
        plt.close()


# overplot all cases
for i_spec in range(n_spec):
    sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]
    spec_name = f"{sv1}_{ar1}x{sv2}_{ar2}"

    for spec in spectra:
        plt.figure(figsize=(12,8))

        for case in cases:

            mean = np.mean(tf[case, spec_name, spec], axis=0)
            std = np.std(tf[case, spec_name, spec], axis=0)
            
            plt.errorbar(lb[id], mean, std, fmt=".", label=f"{spec_name} {spec} {case}")

        plt.legend()
        plt.savefig(f"{plot_dir}/all_case_transfer_function_{spec}_{spec_name}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
