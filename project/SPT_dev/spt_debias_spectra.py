import pylab as plt
import numpy as np
import sys
from copy import deepcopy

from pspy import pspy_utils, so_dict, so_mcm, so_spectra
from pspipe_utils import log, pspipe_list


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

lmax = d["lmax"]
binning_file = d["binning_file"]
release_dir = d["release_dir"]
type = d["type"]

mcm_dir = "mcms"
spec_dir = "spectra"
spec_corr_dir = "spectra_corrected"
pspy_utils.create_directory(spec_corr_dir)


def get_Bbl_dict(Bbl):
    Bbl_dict = {}
    Bbl_dict["tt"] = Bbl["spin0xspin0"]
    Bbl_dict["te"] = Bbl["spin0xspin0"]
    Bbl_dict["et"] = Bbl["spin0xspin0"]
    nbins, my_lmax = int(Bbl["spin2xspin2"].shape[0]/4), int(Bbl["spin2xspin2"].shape[1]/4)
    Bbl_dict["ee"] = Bbl["spin2xspin2"][:nbins, :my_lmax]
    return Bbl_dict

def correct_spt_transfer_function(lb, ps, spec_name, Bbl):

    """
    correct spt transfer function, we bin it using out BBl
    """

    ps_corrected = deepcopy(ps)
    
    tf_dir = f"{release_dir}/ancillary_products/specific_to_c25/"
    
    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")
    
    Bbl_dict = get_Bbl_dict(Bbl)
    
    tf = {}
    for mode in ["tt", "te", "et", "ee"]:
        tf_file = f"{tf_dir}/filter_transfer_function_c25_v1_{fa}ghz{fb}ghz_{mode}.txt"
        l, tf[mode] = np.loadtxt(tf_file, unpack=True)
        tf[mode] = np.dot(Bbl_dict[mode], tf[mode][:lmax])
        ps_corrected[mode.upper()] = ps[mode.upper()] / tf[mode]
        
    return lb, ps_corrected
    
    
def correct_spt_additive_bias(lb, ps, spec_name, Bbl):

    """
    correct spt additive inpainting and filtering bias, note that the bias are given in Dl
    """

    ps_corrected = deepcopy(ps)
    
    tf_dir = f"{release_dir}/ancillary_products/specific_to_c25/"
    
    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")
    
    Bbl_dict = get_Bbl_dict(Bbl)

    additive_bias = {}
    for mode in ["tt", "te", "et", "ee"]:
        filter_artefact_bias_file = f"{tf_dir}/filtering_artifact_bias_{fa}ghz{fb}ghz_{mode}.txt"
        l,  filter_artefact_bias = np.loadtxt(filter_artefact_bias_file, unpack=True)
        inpainting_bias_file = f"{tf_dir}/inpainting_bias_{fa}ghz{fb}ghz_{mode}.txt"
        l,  inpainting_bias = np.loadtxt(inpainting_bias_file, unpack=True)
        additive_bias[mode] = (filter_artefact_bias + inpainting_bias) * l * (l + 1) / (2 * np.pi) #corrections are in Cl
        additive_bias[mode] =  np.dot(Bbl_dict[mode], additive_bias[mode][:lmax])
        ps_corrected[mode.upper()] = ps[mode.upper()] - additive_bias[mode]
        
    return lb, ps_corrected
    
    
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

for spec_name in spec_name_list:
    _, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{spec_name}", spin_pairs=spin_pairs)

    for spec_type in ["auto", "noise", "cross"]:
    
        lb, Db_dict = so_spectra.read_ps(f"spectra/Dl_{spec_name}_{spec_type}.dat", spectra=spectra)
        lb, Db_dict_bias_corr = correct_spt_additive_bias(lb, Db_dict, spec_name, Bbl)
        lb, Db_dict_tf_corr = correct_spt_transfer_function(lb, Db_dict, spec_name, Bbl)
        lb, Db_dict_bias_tf_corr = correct_spt_transfer_function(lb, Db_dict_bias_corr, spec_name, Bbl)

        if spec_type in ["auto", "cross"]:
            # for auto and cross and correct for the additive bias
            # for noise it doesn't make sense
            ps_file = f"{spec_corr_dir}/Dl_{spec_name}_{spec_type}_tf_bias_corr.dat"
            so_spectra.write_ps(ps_file, lb, Db_dict_bias_tf_corr, type, spectra=spectra)

        ps_file = f"{spec_corr_dir}/Dl_{spec_name}_{spec_type}_tf_corr.dat"
        so_spectra.write_ps(ps_file, lb, Db_dict_tf_corr, type, spectra=spectra)

