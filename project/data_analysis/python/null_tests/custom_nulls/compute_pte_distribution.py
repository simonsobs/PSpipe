import pickle
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations_with_replacement as cwr


def get_null_list(surveys, arrays, spectra):
    map_set_list = []
    for sv in surveys:
        for ar in arrays:
            map_set_list.append(f"{sv}_{ar}")
    null_list = []
    for i, (ms1, ms2) in enumerate(cwr(map_set_list, 2)):
        for j, (ms3, ms4) in enumerate(cwr(map_set_list, 2)):
            if j <= i: continue
            for m in spectra:
                null_list += [[m, ms1, ms2, ms3, ms4]]
                
    return null_list
    
def pte_histo(pte_list, file_name, n_bins):
    n_samples = len(pte_list)
    
    bins = np.linspace(0, 1, n_bins + 1)
    
    min, max = np.min(pte_list), np.max(pte_list)
    id_high = np.where(pte_list > 0.99)
    id_low = np.where(pte_list < 0.01)
    
    nPTE_high = len(id_high[0])
    nPTE_low =  len(id_low[0])
    
    plt.figure(figsize=(8,6))
    plt.xlabel(r"Probability to exceed (PTE)")
    plt.hist(pte_list, bins=bins, label=f"n tests: {n_samples}, min: {min:.4f}, max: {max:.4f}, [PTE>0.99]: {nPTE_high},  [PTE<0.01]: {nPTE_low}")
    plt.axhline(n_samples/n_bins, color="k", ls="--")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{file_name}", dpi=300)
    plt.clf()
    plt.close()
    
    
null_test = "inout"
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


if null_test == "elevation":
    surveys = ["el1", "el2", "el3"]
    test_arrays = [["pa4_f220"], ["pa5_f090"], ["pa5_f150"],  ["pa6_f090"], ["pa6_f150"]]
    n_bins = 30
    
if null_test == "pwv":
    surveys = ["pwv1", "pwv2"]
    test_arrays = [["pa4_f220"], ["pa5_f090"], ["pa5_f150"],  ["pa6_f090"], ["pa6_f150"]]
    n_bins = 10
    
if null_test == "inout":
    surveys = ["dr6"]
    test_arrays = [["pa4_f220_in", "pa4_f220_out"],
                   ["pa5_f090_in", "pa5_f090_out"],
                   ["pa5_f150_in", "pa5_f150_out"],
                   ["pa6_f090_in", "pa6_f090_out"],
                   ["pa6_f150_in", "pa6_f150_out"]]
    n_bins = 10

pte_list = []
for t_ar in test_arrays:
    null_list = get_null_list(surveys, t_ar, spectra)
    for null in null_list:
        mode, ms1, ms2, ms3, ms4 = null
        
        if null_test in ["elevation", "pwv"]:
            my_ar = t_ar[0]
        else:
            my_ar = t_ar[0][:8]
        
        if (ms1 == ms2) & (ms3 == ms4) & (mode in ["ET", "BT", "BE"]) :
            print(f"skip {ms1}x{ms2}- {ms3}x{ms4} {mode} since it's a doublon of {mode[::-1]}")
            continue
        if (my_ar == "pa4_f220" ) & (mode != "TT"):
            continue
        
        res_dict = pickle.load(open(f"{null_test}/null_test_{my_ar}/diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}.pkl", "rb"))

        name = res_dict["fname"]
        chi2 = res_dict["chi2"]
        pte = res_dict["pte"]

        pte_list = np.append(pte_list, pte)

pte_histo(pte_list, null_test, n_bins)
