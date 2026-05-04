"""
This script performs array null tests and plot residual power spectra and a summary PTE distribution
"""

from pspy import so_dict, pspy_utils, so_cov, so_spectra, so_mpi
from pspipe_utils import consistency, best_fits, log, pspipe_list
import pickle
import scipy.stats as ss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import yaml

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
log = log.get_logger(**d)

with open(d['nulls_yaml'], "r") as f:
    infos_dict: dict = yaml.safe_load(f)
null_infos = infos_dict['compute_null_tests.py']

pte_threshold = null_infos['pte_threshold']
remove_first_bins = null_infos['remove_first_bins']
skip_diff_freq_TT = null_infos['skip_diff_freq_TT']
skip_EB = null_infos['skip_EB']
fudge = null_infos['fudge']
plot_all = True

multipole_range = null_infos['multipole_range']     # This was previously in null_info.py
l_pows = null_infos['l_pows']

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
hist_label = ""
if fudge == True:
    hist_label += "_fudge"
if skip_EB == True:
    hist_label += "_skip_EB"
if skip_diff_freq_TT == True:
    hist_label += "_skip_TT_diff_freq"

spectra_dir = d["spec_dir"]
cov_dir = d["cov_dir"]
plot_dir = d["plots_dir"] + '/nulls/'
bestfits_dir = d['best_fits_dir']

null_test_dir = d["nulls_dir"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

pspy_utils.create_directory(null_test_dir)
pspy_utils.create_directory(plot_dir)
for spec in spectra:
    pspy_utils.create_directory(plot_dir + f'{spec}/')
test_infos = null_infos['test_infos']   # {name: {spec_dir:..., cov_dir:..., cov_correction:...}, name2:{...}}

null_list = pspipe_list.get_null_list_from_cov_list(d, spectra=spectra, remove_TT_diff_freq=True)
n_nulls = len(null_list)
spec_list = pspipe_list.get_spectra_list(d)
all_ps = {}
all_cov = {}
chi2_dict = {}
for name, infos in test_infos.items():
    cov_template = f"{infos['cov_dir']}" + "/analytic_cov_{}x{}_{}x{}.npy"
    ps_template = f"{infos['spec_dir']}" + "/Dl_{}x{}" + f"_cross.dat"
    all_ps[name], all_cov[name] = consistency.get_nulls_ps_and_cov_dict(null_list, ps_template, cov_template, spectra_order=spectra)
    lb = all_ps[name]["ell"]
# all_ps is a dict such as all_ps = {test_name: {(sv_ar1, sv_ar2, spec): 1D np.ndarray}}
# all_cov is a dict such as all_cov = {test_name: {(sv_ar1, sv_ar2, spec12), (sv_ar3, sv_ar4, spec34): 2D np.ndarray}}

# Load foreground best fits
fg_file_name = f"{bestfits_dir}" + "fg_{}x{}.dat"
l_fg, fg_dict = best_fits.fg_dict_from_files_and_spec_list(fg_file_name, spec_list, d["lmax"], spectra=spectra)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_nulls - 1)
log.info(f"[Rank {so_mpi.rank}] number of cross-map pairs to compute: {len(subtasks)} / {n_nulls}")

for task in subtasks:
    null = null_list[task]
    
    mode, ms1, ms2, ms3, ms4 = null

    lmin, lmax = {}, {}

    res_ps, res_cov = {}, {}

    for name, infos in test_infos.items():
        lmin[name], lmax[name] = get_lmin_lmax(null, multipole_range[name])
        lb, res_ps[name], res_cov[name],  = consistency.compare_spectra([ms1, ms2, ms3, ms4],
                                                                "ab-cd",
                                                                all_ps[name],
                                                                all_cov[name],
                                                                mode = mode,
                                                                return_chi2=False)
        
        if remove_first_bins:
            lb, res_ps[name], res_cov[name] = lb[remove_first_bins:], res_ps[name][remove_first_bins:], res_cov[name][remove_first_bins:,remove_first_bins:]

    res_fg = fg_dict[ms1, ms2][mode] - fg_dict[ms3, ms4][mode]
    lb_fg, res_fg_b = pspy_utils.naive_binning(l_fg, res_fg, d["binning_file"], d["lmax"])
    
    if remove_first_bins:
        lb_fg, res_fg_b = lb_fg[remove_first_bins:], res_fg_b[remove_first_bins:]

    fname = f"diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}"

    # ZACH YOU MAY WANT TO UNCOMMENT THAT FOR THE INTERACTIVE NULLS TABLE
    # for name, test in test_infos.items():
    #     spec_dir = test["spec_dir"]
    #     cov_correction = test["cov_correction"]
        
    #     sigma = np.sqrt(np.diagonal(res_cov[name]))
        
    #     np.save(f"{null_test_dir}/ps_{label}_{fname}.npy", res_ps[name] )
    #     np.savetxt(f"{null_test_dir}/cov_{label}_{fname}.npy", np.transpose([lb, res_cov[name], sigma]))

    # Plot residual and get chi2
    plot_title = f"{ms1}x{ms2} - {ms3}x{ms4}"
    expected_res = 0.
    
    lb_th, res_th = lb_fg, res_fg_b
    assert len(lb) == len(lb_th), "Mismatch between expected residual and data"

    pte_list = []
    for i, (name, infos) in enumerate(test_infos.items()):
        lrange = np.where((lb >= lmin[name]) & (lb <= lmax[name]))[0]
        chi2 = (res_ps[name][lrange] - res_th[lrange]) @ np.linalg.inv(res_cov[name][np.ix_(lrange, lrange)]) @ (res_ps[name][lrange] - res_th[lrange])
        ndof = len(lb[lrange])

        pte = 1 - ss.chi2(ndof).cdf(chi2)
        chi2_dict[name, tuple(null)] = {"chi2": chi2, "ndof": ndof, "pte" : pte}
        pte_list.append(pte)

    if plot_all:
        colors = ["navy", "red", "forestgreen", "darkorange"]
        fig, ax = plt.subplots(2, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios':(2, 1), "hspace": 0}, dpi=200)
        ellshift = np.linspace(-3, 3, len(test_infos))
        for i, (name, infos) in enumerate(test_infos.items()):
            ax[0].errorbar(lb+ellshift[i], res_ps[name] * lb ** l_pows[mode],
                yerr=np.sqrt(res_cov[name].diagonal()) * lb ** l_pows[mode],
                ls="None", marker = ".", ecolor = colors[i],
                color=colors[i], alpha=.8,
                label=fr"{name} [$\chi^2 = {{{chi2_dict[name, tuple(null)]["chi2"]:.1f}}}/{{{chi2_dict[name, tuple(null)]["ndof"]}}}$ (${{{chi2_dict[name, tuple(null)]["pte"]:.4f}}}$)]"
            )
            ax[1].errorbar(lb+ellshift[i], res_ps[name] / np.sqrt(res_cov[name].diagonal()),
                yerr=np.ones_like(lb),
                ls="None", marker = ".", ecolor = colors[i],
                color=colors[i], alpha=.8,
            )
            for a in range(2):
                ax[a].axvspan(xmin=0, xmax=lmin[name],
                        color=colors[i], alpha=0.15)
                ax[a].axvspan(xmin=lmax[name], xmax=10000,
                        color=colors[i], alpha=0.15)

        for a in range(2):
            ax[a].axhline(0., color='black', ls='--', zorder=-10)
        
        ax[0].set_xlim(50, d["lmax"])
        ax[0].legend()
        plt.savefig(f"{plot_dir}/{mode}/{fname}.png")
        plt.close()
    pte_comment = '!' * int(-np.log10(min(min(pte_list), 1-max(pte_list)))) # I love this line
    log.info(f"[Rank {so_mpi.rank}] {" ".join(null)} PTEs: {" ".join([f"{pte:.5f}" for pte in pte_list])} {pte_comment}")


so_mpi.barrier()
chi2_dict = so_mpi.gather_set_or_dict(chi2_dict, allgather=False,
                                                root=0, overlap_allowed=False)

if so_mpi.rank == 0:

    # create pte dicts and filter given combinations and duplicates for plot
    pte_dict = {}
    for (name, (mode, ms1, ms2, ms3, ms4)), chi2_ndof_pte in chi2_dict.items():
        pte_dict[(name, (mode, ms1, ms2, ms3, ms4))] = chi2_ndof_pte["pte"]


    # Save pte to pickle
    with open(f"{plot_dir}/chi2_dict.pkl", "wb") as f:
        pickle.dump(chi2_dict, f)
    with open(f"{plot_dir}/pte_dict.pkl", "wb") as f:
        pickle.dump(pte_dict, f)

    if skip_EB == True:
        tested_spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "BB"]
    else:
        tested_spectra = spectra

    # Plot PTE histogram for each cov type
    for name in test_infos.keys():
        pte_list = pte_dict[name, 'all']
        log.info(f"{name} Worse PTE: min {np.min(pte_list)}, max {np.max(pte_list)}")
        n_bins = 14
        file_name = f"{plot_dir}/pte_hist_all_{name}{hist_label}.png"
        pte_histo(pte_list, file_name, n_bins)
        for mode in tested_spectra:
            pte_list = pte_dict[name, mode]
            log.info(f"{len(pte_list)} test for mode {mode}")
            n_bins = 8
            file_name = f"{plot_dir}/pte_hist_{mode}_{name}{hist_label}.png"
            pte_histo(pte_list, file_name, n_bins)

