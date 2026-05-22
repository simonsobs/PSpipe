import pylab as plt
import numpy as np
import sys
from copy import deepcopy

from pspy import pspy_utils, so_dict, so_mcm, so_spectra
from pspipe_utils import log, pspipe_list


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


spec_dir = "spectra"
spec_corr_dir = "spectra_corrected_pspipe"
tf_dir = "transfer_functions"

pspy_utils.create_directory(spec_corr_dir)

type = d["type"]

    
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

for spec_name in spec_name_list:

    tf = {}
    for spec in ["TT", "EE", "BB"]:
        lb_tf, tf[spec], sigma_tf = np.loadtxt(f"{tf_dir}/transfer_function_filter_masking_yes_alm_mask_{spec}_{spec_name}.dat", unpack=True)
    
    for spec in spectra:
        a, b = spec
        if a != b:
            print(a, b, "ok")
            tf[spec] = np.sqrt(tf[a + a] * tf[b + b])
            
    for spec_type in ["auto", "noise", "cross"]:
    
        lb, Db_dict = so_spectra.read_ps(f"spectra/{type}_{spec_name}_{spec_type}.dat", spectra=spectra)
        id = np.where(lb >= lb_tf[0]) # use the same lmin
        
        ps_dict = {}
        for spec in spectra:
            ps_dict[spec] = Db_dict[spec][id] / tf[spec]
        
        so_spectra.write_ps(spec_corr_dir + f"/{type}_{spec_name}_{spec_type}.dat", lb[id], ps_dict, type, spectra=spectra)
