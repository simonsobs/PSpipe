"""
script to help making the calibration table to run it:
python AxP_cal_table.py global_dr6v4xlegacy_updated.dict global_dr6v4xnpipe_updated.dict
you have to have computed both npipe and legacy numbers
"""
from pspy import so_dict
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle

d_legacy = so_dict.so_dict()
d_legacy.read_from_file(sys.argv[1])

d_npipe = so_dict.so_dict()
d_npipe.read_from_file(sys.argv[2])

arrays = ["pa5_f090", "pa6_f090", "pa5_f150", "pa6_f150", "pa4_f220"]
methods = ["AxA-AxP", "AxA-PxP", "PxP-AxP"]
cal_folder = "calibration_results_planck_bias_corrected_fg_sub"

with open(f"dr6xlegacy/{cal_folder}/calibs_dict.pkl", "rb") as f:
    cal_legacy = pickle.load(f)

with open(f"dr6xnpipe/{cal_folder}/calibs_dict.pkl", "rb") as f:
    cal_npipe = pickle.load(f)

# These are the numbers applied to the released maps
map_cal = {}
map_cal["pa5_f090"] = 1.0111
map_cal["pa6_f090"] = 1.0086
map_cal["pa5_f150"] = 0.9861
map_cal["pa6_f150"] = 0.9702
map_cal["pa4_f220"] = 1.0435

for ar in arrays:
    str = ""
    for method in methods:
        cal, sigma = cal_legacy[method, ar]["calibs"]
        cal *= d_legacy[f"cal_dr6_{ar}"] * map_cal[ar]
        str += f"& {cal:.4f} $\pm$ {sigma:.4f} "
    for method in methods:
        cal, sigma = cal_npipe[method, ar]["calibs"]
        cal *= d_npipe[f"cal_dr6_{ar}"] * map_cal[ar]
        str += f"& {cal:.4f} $\pm$ {sigma:.4f} "
        
    waf, f = ar.split("_")
    print(f"{f} & {waf.upper()} {str}& {map_cal[ar]} \\")
