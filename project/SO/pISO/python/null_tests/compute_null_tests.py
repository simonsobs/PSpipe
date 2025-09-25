"""
This script performs array null tests and plot residual power spectra and a summary PTE distribution
"""

from pspy import so_dict, pspy_utils, so_cov, so_spectra
from pspipe_utils import consistency, best_fits, covariance, pspipe_list
import pickle
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
from null_infos import *

def get_lmin_lmax(null, multipole_range):
    """
    compute the lmin and lmax associated to a given null test
    """
    m, ar1, ar2, ar3, ar4 = null
    m0, m1 = m[0], m[1]
    lmin0, lmax0 = multipole_range[ar1][m0]
    lmin1, lmax1 = multipole_range[ar2][m1]
    ps12_lmin = max(lmin0, lmin1)
    ps12_lmax = min(lmax0, lmax1)
    lmin2, lmax2 = multipole_range[ar3][m0]
    lmin3, lmax3 = multipole_range[ar4][m1]
    ps34_lmin = max(lmin2, lmin3)
    ps34_lmax = min(lmax2, lmax3)
    lmin = max(ps12_lmin, ps34_lmin)
    lmax = min(ps12_lmax, ps34_lmax)
    
    return lmin, lmax

def pte_histo(pte_list, file_name, n_bins):
    n_samples = len(pte_list)
    bins = np.linspace(0, 1, n_bins + 1)
    min, max = np.min(pte_list), np.max(pte_list)
    
    plt.figure(figsize=(8,6))
    plt.title("Array-bands test", fontsize=16)
    plt.xlabel(r"Probability to exceed (PTE)", fontsize=16)
    plt.hist(pte_list, bins=bins, label=f"n tests: {n_samples}, min: {min:.3f}, max: {max:.3f}", histtype="bar", facecolor="orange", edgecolor="black", linewidth=3)
    plt.axhline(n_samples/n_bins, color="k", ls="--", alpha=0.5)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.savefig(f"{file_name}", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    
def check_freq_pair(f1, f2, f3, f4):
    count = 0
    if (f1 == f3) & (f2 == f4):
        count += 1
    if (f1 == f4) & (f2 == f3):
        count += 1
    if count != 0:
        return True
    else:
        return False

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

pte_threshold = 0.01
remove_first_bin = True

skip_pa4_pol = True
skip_diff_freq_TT = True
skip_EB = False
fudge = False

plot_dir = "plots/array_nulls"
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
hist_label = ""
if skip_pa4_pol == True:
    hist_label += "skip_pa4pol"
if fudge == True:
    hist_label += "_fudge"
if skip_EB == True:
    hist_label += "_skip_EB"
if skip_diff_freq_TT == True:
    hist_label += "_skip_TT_diff_freq"



cov_dir = "covariances"
null_test_dir = "null_test"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

pspy_utils.create_directory(null_test_dir)
pspy_utils.create_directory(plot_dir)

test_list = [{"name":  "spectra_corrected+mc_cov+beam_cov+leakage_cov",
              "spec_dir": "spectra_leak_corr",
              "cov_correction": ["mc_cov", "beam_cov","leakage_cov"]}]
               

spec_dir_list = []
cov_type_list = ["analytic_cov"]
label_list = []
for test in test_list:
    spec_dir_list = np.append(spec_dir_list, test["spec_dir"])
    cov_type_list = np.append(cov_type_list, test["cov_correction"])
    label_list += [test["name"]]

cov_type_list = list(dict.fromkeys(cov_type_list)) #remove doublon
spec_dir_list = list(dict.fromkeys(spec_dir_list)) #remove doublon
map_set_list = pspipe_list.get_map_set_list(d)


all_cov = {}
_ps_temp = "spectra" + "/Dl_{}x{}_cross.dat"
for cov in cov_type_list:
    cov_template = f"{cov_dir}/{cov}" + "_{}x{}_{}x{}.npy"
    _, all_cov[cov] =  consistency.get_ps_and_cov_dict(map_set_list, _ps_temp, cov_template, spectra_order=spectra)
    
all_ps = {}
for spec_dir in spec_dir_list:
    ps_template = spec_dir + "/Dl_{}x{}_cross.dat"
    all_ps[spec_dir], _ = consistency.get_ps_and_cov_dict(map_set_list, ps_template, cov_template, spectra_order=spectra)
    lb = all_ps[spec_dir]["ell"]


# Load foreground best fits
fg_file_name = "best_fits/fg_{}x{}.dat"
l_fg, fg_dict = best_fits.fg_dict_from_files(fg_file_name, map_set_list, d["lmax"], spectra=spectra)

# Define PTE dict
pte_dict = {}
for label in label_list:
    pte_dict[label, "all"] = []
    for spec in spectra:
        pte_dict[label, spec] = []
    
operations = {"diff": "ab-cd"}

null_list = pspipe_list.get_null_list(d, spectra=spectra, remove_TT_diff_freq=False)

for null in null_list:

        
    mode, ms1, ms2, ms3, ms4 = null
    lmin, lmax = get_lmin_lmax(null, multipole_range)

    res_ps, res_cov = {}, {}

    for spec_dir in spec_dir_list:
        lb, res_ps[spec_dir], _,  = consistency.compare_spectra([ms1, ms2, ms3, ms4],
                                                                "ab-cd",
                                                                all_ps[spec_dir],
                                                                all_cov["analytic_cov"],
                                                                mode = mode,
                                                                return_chi2=False)
        if remove_first_bin:
            res_ps[spec_dir] = res_ps[spec_dir][1:]
            
            
    for cov in cov_type_list:
        lb, _, res_cov[cov] = consistency.compare_spectra([ms1, ms2, ms3, ms4],
                                                          "ab-cd",
                                                          all_ps[spec_dir_list[0]],
                                                          all_cov[cov],
                                                          mode = mode,
                                                          return_chi2=False)
        if remove_first_bin:
            lb, res_cov[cov] = lb[1:], res_cov[cov][1:,1:]
            

    res_fg = fg_dict[ms1, ms2][mode] - fg_dict[ms3, ms4][mode]
    lb_fg, res_fg_b = pspy_utils.naive_binning(l_fg, res_fg, d["binning_file"], d["lmax"])
    
    if remove_first_bin:
        lb_fg, res_fg_b = lb_fg[1:], res_fg_b[1:]

    
    res_ps_dict = {}
    res_cov_dict = {}
    
    fname = f"diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}"

    for test in test_list:
        spec_dir = test["spec_dir"]
        cov_correction = test["cov_correction"]
        label = test["name"]


        res_ps_dict[label] = res_ps[spec_dir]
        res_cov_dict[label] = res_cov["analytic_cov"]
        
        if "mc_cov" in cov_correction:
            res_cov_dict[label] = covariance.correct_analytical_cov(res_cov_dict[label],
                                                                    res_cov["mc_cov"],
                                                                    only_diag_corrections=True,
                                                                    use_max_error=False)
        if "beam_cov" in cov_correction:
            res_cov_dict[label] += res_cov["beam_cov"]
            
        if "leakage_cov" in cov_correction:
            res_cov_dict[label] += res_cov["leakage_cov"]
            
        if fudge:
            ind = np.diag_indices_from(res_cov_dict[label])
            print("WARNING FUDGE FACTOR")
            fudge_error = 1.02
            res_cov_dict[label][ind] *= fudge_error ** 2
        
        sigma = np.sqrt(np.diagonal(res_cov_dict[label]))
        

        np.save(f"{null_test_dir}/ps_{label}_{fname}.npy", res_cov_dict[label] )
        np.savetxt(f"{null_test_dir}/cov_{label}_{fname}.npy", np.transpose([lb, res_ps_dict[label], sigma]))

    
    # Plot residual and get chi2
    lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
    plot_title = f"{ms1}x{ms2} - {ms3}x{ms4}"
    expected_res = 0.
    
    
    ylims = y_lims[mode]
    if "pa4" in fname:
        ylims = (y_lims[mode][0] * 10,  y_lims[mode][1] * 10)

    chi2_dict = consistency.plot_residual(lb, res_ps_dict, res_cov_dict, mode=mode,
                                          title=plot_title.replace("dr6_", ""),
                                          file_name=f"{plot_dir}/{fname}",
                                          expected_res=expected_res,
                                          lrange=lrange,
                                          overplot_theory_lines=(lb_fg, res_fg_b),
                                          l_pow=l_pows[mode],
                                          return_chi2=True,
                                          ylims=ylims)
                                          
                 
    
    if skip_pa4_pol == True:
        if ("pa4" in fname) & (mode != "TT"):
            print(f"skip {ms1}x{ms2}- {ms3}x{ms4} {mode}, we don't use pa4 in pol")

            continue
    if (skip_EB == True) & (mode in ["EB", "BE"]):
        print(f"skip {ms1}x{ms2}- {ms3}x{ms4} {mode}, EB is used as a test of the polarisation angle")
        continue
    if (ms1 == ms2) & (ms3 == ms4) & (mode in ["ET", "BT", "BE"]) :
        print(f"skip {ms1}x{ms2}- {ms3}x{ms4} {mode} since it's a doublon of {mode[::-1]}")
        continue
    f1, f2 = d[f"freq_info_{ms1}"]["freq_tag"],  d[f"freq_info_{ms2}"]["freq_tag"]
    f3, f4 = d[f"freq_info_{ms3}"]["freq_tag"],  d[f"freq_info_{ms4}"]["freq_tag"]
    if (mode == "TT") & (skip_diff_freq_TT == True) & (check_freq_pair(f1, f2, f3, f4) == False):
        print(f"skip {ms1}x{ms2}- {ms3}x{ms4} {mode} since TT has different frequencies")

        continue
    
    # Fill pte_dict
    for label in label_list:
        pte = 1-ss.chi2(chi2_dict[label]["ndof"]).cdf(chi2_dict[label]["chi2"])
        pte_dict[label, mode].append(pte)
        pte_dict[label, "all"].append(pte)

        if (pte <= pte_threshold) or (pte >= 1-pte_threshold):
            print(f"[{label}] [{plot_title} {mode}] PTE = {pte:.04f}")


# Save pte to pickle
pickle.dump(pte_dict, open(f"{plot_dir}/pte_dict.pkl", "wb"))
  
if skip_EB == True:
    tested_spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "BB"]
else:
    tested_spectra = spectra

# Plot PTE histogram for each cov type
for label in label_list:
    pte_list = pte_dict[label, "all"]
    print(f"{label} Worse PTE: min {np.min(pte_list)}, max {np.max(pte_list)}")
    n_bins = 14
    file_name = f"{plot_dir}/pte_hist_all_{label}{hist_label}.png"
    pte_histo(pte_list, file_name, n_bins)
    for mode in tested_spectra:
        pte_list = pte_dict[label, mode]
        print(f"{len(pte_list)} test for mode {mode}")
        n_bins = 8
        file_name = f"{plot_dir}/pte_hist_{mode}_{label}{hist_label}.png"
        pte_histo(pte_list, file_name, n_bins)

