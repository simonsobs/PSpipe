"""
This script performs null tests
and plot residual power spectra and a summary PTE distribution
"""

from pspy import so_dict, pspy_utils, so_cov, so_spectra
from pspipe_utils import consistency, best_fits, covariance, pspipe_list
import pickle
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
from null_infos import *
import pickle

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

    plt.figure(figsize=(8,6))
    plt.xlabel(r"Probability to exceed (PTE)")
    plt.hist(pte_list, bins=bins)
    plt.axhline(n_samples/n_bins, color="k", ls="--")
    plt.tight_layout()
    plt.savefig(f"{file_name}", dpi=300)
    plt.clf()
    plt.close()

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

fudge = False
pte_threshold = 0.01
remove_first_bin = True
skip_EB = False
create_slides = True


cov_dir = "covariances"
null_test_dir = "null_test"
plot_dir = "plots/array_nulls"


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if skip_EB == True:
    tested_spectra = ["TT", "TE", "ET", "TB", "BT", "EE", "BB"]
    plot_dir = "plots/array_nulls_skip_EB"
else:
    tested_spectra = spectra
    plot_dir = "plots/array_nulls"


pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(null_test_dir)

test_list = [{"name":  "spectra+beam+mc_cov",
              "spec_dir": "spectra",
              "cov_correction": ["mc_cov", "beam_cov"]}]
               

test_list += [{"name": "spectra_with_analytic_cov",
               "spec_dir": "spectra",
               "cov_correction": []}]

# we start by making list of all the elements we need to load in memory
# for this we loop on the differnet test and identify which elements are required for them
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

# we now read all covariances, note that the function call is a bit annoying
# this is because the function is supposed to load spectra and cov at the same time

all_cov = {}
_ps_temp = "spectra" + "/Dl_{}x{}_cross.dat"
for cov in cov_type_list:
    cov_template = f"{cov_dir}/{cov}" + "_{}x{}_{}x{}.npy"
    _, all_cov[cov] =  consistency.get_ps_and_cov_dict(map_set_list, _ps_temp, cov_template, spectra_order=spectra)
    
# now I'm reading all spectra

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
    for spec in tested_spectra:
        pte_dict[label, spec] = []
    
operations = {"diff": "ab-cd"}

null_list = pspipe_list.get_null_list(d, spectra=tested_spectra) #skip_EB


print(label_list)




for null in null_list:

        
    mode, ms1, ms2, ms3, ms4 = null
    lmin, lmax = get_lmin_lmax(null, multipole_range)

    res_ps, res_cov = {}, {}

    # same comment as before in the current implementation residual ps and cov
    # are read at the same time while for us we prefer to read ps and cov separately, so the ugly hack
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
                                                          all_ps["spectra"],
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
        
        #plt.plot(res_cov_dict[label].diagonal())
        #plt.plot(res_cov["mc_cov"].diagonal())
        #plt.show()
        if "mc_cov" in cov_correction:
            res_cov_dict[label] = covariance.correct_analytical_cov(res_cov_dict[label],
                                                                    res_cov["mc_cov"],
                                                                    only_diag_corrections=True)
        if "beam_cov" in cov_correction:
            res_cov_dict[label] += res_cov["beam_cov"]
            
        if "leakage_cov" in cov_correction:
            res_cov_dict[label] += res_cov["leakage_cov"]
            
        
        sigma = np.sqrt(np.diagonal(res_cov_dict[label]))
        

    
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
                                          

    save_label = "spectra+beam+mc_cov"
    res_dict = {
            "fname" :  f"diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}",
            "ell": lb,
            "lrange": lrange,
            "res_ps": res_ps_dict[save_label],
            "res_cov": res_cov_dict[save_label],
            "res_fg": res_fg,
            "chi2": chi2_dict[save_label]["chi2"],
            "ndof": chi2_dict[save_label]["ndof"],
            "pte": 1-ss.chi2(chi2_dict[save_label]["ndof"]).cdf(chi2_dict[save_label]["chi2"])
            }
    pickle.dump(res_dict, open(f"{null_test_dir}/{fname}.pkl", "wb"))


    if (ms1 == ms2) & (ms3 == ms4) & (mode in ["ET", "BT", "BE"]) :
        print("")
        print(f"skip {ms1}x{ms2}- {ms3}x{ms4} {mode} since it's a doublon of {mode[::-1]}")
        print("")

        continue
    else:
        # Fill pte_dict
        for label in label_list:
            pte = 1-ss.chi2(chi2_dict[label]["ndof"]).cdf(chi2_dict[label]["chi2"])
            pte_dict[label, mode].append(pte)
            pte_dict[label, "all"].append(pte)

            if (pte <= pte_threshold) or (pte >= 1-pte_threshold):
                print(f"[{label}] [{plot_title} {mode}] PTE = {pte:.04f}")


n_null = len(pte_dict["spectra_with_analytic_cov", "all"])

print(f"we have done {n_null} null tests")

# Save pte to pickle
pickle.dump(pte_dict, open(f"{plot_dir}/pte_dict.pkl", "wb"))

# Plot PTE histogram for each cov type
for label in label_list:
    pte_list = pte_dict[label, "all"]
    print(f"{label} Worse PTE: min {np.min(pte_list)}, max {np.max(pte_list)}")
    n_bins = 10
    file_name = f"{plot_dir}/pte_hist_all_{label}.png"
    pte_histo(pte_list, file_name, n_bins)
    for mode in tested_spectra:
        pte_list = pte_dict[label, mode]
        print(f"{len(pte_list)} test for mode {mode}")
        n_bins = 8
        file_name = f"{plot_dir}/pte_hist_{mode}_{label}.png"
        pte_histo(pte_list, file_name, n_bins)

