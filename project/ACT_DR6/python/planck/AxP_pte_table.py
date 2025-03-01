"""
script to help making the pte table to run it:
you have to have ran AxP_residuals.py
"""
from pspy import so_dict
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle

plot_dir = "paper_plot"

with open(f"{plot_dir}/pte_dict.pkl", "rb") as f:
    pte_dict = pickle.load(f)

spectra = ["EE", "ET"]
runs = ["legacy", "NPIPE"]
combination = ["AxA-PxP", "AxP-PxP"]

dr6 = ["dr6_pa5_f090", "dr6_pa5_f150", "dr6_pa6_f090", "dr6_pa6_f150"]
planck = ["Planck_f100", "Planck_f143", "Planck_f217"]
for spec, comb in zip(spectra, combination):
    for ar_act in dr6:
    
        ar_name = ar_act.replace("dr6_", "")
        waf, freq = ar_name.split("_")
        ar_name = f"{waf.upper()}~{freq}"
        
        my_str = ar_name
        for ar_planck in planck:

            if comb == "AxA-PxP":
                test = f"{ar_act}x{ar_act} - {ar_planck}x{ar_planck}"
            else:
                test = f"{ar_act}x{ar_planck} - {ar_planck}x{ar_planck}"
            
            for run in runs:
                pte = pte_dict[spec, comb, run][test]
                my_str += f" & {pte * 100:.0f} \% "
        
        my_str +=  "\\"
        my_str +=  "\\"

        print(my_str)
    if spec == "EE":
        print("\hline")
        print("\hline")
