"""
look at the PTE correlation of the ACT vs Planck test
"""
from pspy import so_dict, pspy_utils, so_cov, so_spectra
from pspipe_utils import consistency, best_fits, covariance, pspipe_list
import pickle
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
import AxP_utils
import matplotlib

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
remove_first_bin = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "20"
n_sims = 300

legacy_dir = "dr6xlegacy/"
sim_spec_dir = f"{legacy_dir}/sim_spectra_legacy"
cov_dir = f"{legacy_dir}/covariances"
spec_dir = f"{legacy_dir}/spectra_leak_corr_planck_bias_corr"
bf_dir = f"{legacy_dir}/best_fits"


plot_dir = "paper_plot"
pte_dir = "PTE_AxP"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(pte_dir)


tested_spectra = [["EE","AxA-PxP"], ["ET","AxP-PxP"]]
null_list = {}
# We don't do TT since it's wrong to use the fg model without propagating uncertainties


null_list = {}
null_list["EE", "AxA-PxP"] = []
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f100", "Planck_f100"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f100", "Planck_f100"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f100", "Planck_f100"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f100", "Planck_f100"]]

null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f143", "Planck_f143"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f143", "Planck_f143"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f143", "Planck_f143"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f143", "Planck_f143"]]

null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f217", "Planck_f217"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f217", "Planck_f217"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f217", "Planck_f217"]]
null_list["EE", "AxA-PxP"] += [["EE", "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f217", "Planck_f217"]]

null_list["ET", "AxP-PxP"] = []
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa5_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa5_f150", "Planck_f100", "Planck_f100", "Planck_f100"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa6_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa6_f150", "Planck_f100", "Planck_f100", "Planck_f100"]]

null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa5_f090", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa5_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa6_f090", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa6_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]

null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa5_f090", "Planck_f217", "Planck_f217", "Planck_f217"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa5_f150", "Planck_f217", "Planck_f217", "Planck_f217"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa6_f090", "Planck_f217", "Planck_f217", "Planck_f217"]]
null_list["ET", "AxP-PxP"] += [["ET", "dr6_pa6_f150", "Planck_f217", "Planck_f217", "Planck_f217"]]


cov_type_list = ["analytic_cov", "mc_cov"]

map_set_list = pspipe_list.get_map_set_list(d)

lb, all_ps, all_cov = AxP_utils.read_data(map_set_list, spec_dir, cov_dir, cov_type_list, spectra)

# Load foreground best fits
fg_file_name = bf_dir + "/fg_{}x{}.dat"
l_fg, fg_dict = best_fits.fg_dict_from_files(fg_file_name, map_set_list, d["lmax"], spectra=spectra)


multipole_range, _, _ = AxP_utils.get_plot_params()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

for test_id, test in enumerate(tested_spectra):
    spec, comb  = test
    
    n_null = len(null_list[spec, comb])
    pte_array = np.zeros((n_null, n_sims))

    null_name_list = []

    for count, null in enumerate(null_list[spec, comb]):
        mode, ms1, ms2, ms3, ms4 = null
        
        null_name = f"{ms1} - {ms3}"
        null_name = null_name.replace("dr6_", "")
        null_name = null_name.replace("pa", "PA")
        null_name = null_name.replace("Planck_f", "Planck f")
        null_name = null_name.replace("_", " ")
        null_name_list += [null_name]
            
        res_fg = fg_dict[ms1, ms2][mode] - fg_dict[ms3, ms4][mode]
        lb_fg, res_fg_b = pspy_utils.naive_binning(l_fg, res_fg, d["binning_file"], d["lmax"])
    
        if remove_first_bin:
            lb_fg, res_fg_b = lb_fg[1:], res_fg_b[1:]

            
        res_cov_dict = {}
        # get the covariance
        for cov in cov_type_list:
            lb, _, res_cov_dict[cov] = consistency.compare_spectra([ms1, ms2, ms3, ms4],
                                                                    "ab-cd",
                                                                    all_ps,
                                                                    all_cov[cov],
                                                                    mode = mode,
                                                                    return_chi2=False)
                                                                    

            if remove_first_bin:
                lb, res_cov_dict[cov] = lb[1:], res_cov_dict[cov][1:,1:]

        r_cov = covariance.correct_analytical_cov(res_cov_dict["analytic_cov"],
                                                  res_cov_dict["mc_cov"],
                                                  only_diag_corrections=True)
                                                  
        lmin, lmax = AxP_utils.get_lmin_lmax(null, multipole_range)

        # Plot residual and get chi2
        lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
        ndof = len(lb[lrange])

        pte_list = []
        for iii in range(n_sims):
        
            _, ps_1 = so_spectra.read_ps(f"{sim_spec_dir}/Dl_{ms1}x{ms2}_cross_{iii:05d}.dat", spectra=spectra)
            _, ps_2 = so_spectra.read_ps(f"{sim_spec_dir}/Dl_{ms3}x{ms4}_cross_{iii:05d}.dat", spectra=spectra)

            res_ps = ps_1[mode] - ps_2[mode]

            if remove_first_bin:
                res_ps = res_ps[1:]

            chi2_sim = (res_ps[lrange] - res_fg_b[lrange]) @ np.linalg.inv(r_cov[np.ix_(lrange, lrange)]) @ (res_ps[lrange] - res_fg_b[lrange])

            pte = 1 - ss.chi2(ndof).cdf(chi2_sim)

            pte_array[count, iii] = pte
            
    np.savetxt(f"{pte_dir}/pte_array.dat", pte_array)
    cov_pte = np.cov(pte_array)
    corr_pte = so_cov.cov2corr(cov_pte)
    
    ax = axes.flat[test_id]
    im = ax.imshow(corr_pte, cmap="seismic", vmin=-0.1, vmax=1)
    ax.set_title(f"{spec[::-1]} (ACT - Planck)", fontsize=24)
    ax.set_xticks(np.arange(n_null), null_name_list, rotation=90)
    if test_id == 0:
        ax.set_yticks(np.arange(n_null), null_name_list)
    else:
        ax.set_yticks([])

fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022)#, pad=1.4)
plt.savefig(f"{plot_dir}/all_PTE_corr.pdf", bbox_inches='tight')
plt.clf()
plt.close()


for test_id, test in enumerate(tested_spectra):
    spec, comb  = test
    for count, null in enumerate(null_list[spec, comb]):
        mode, ms1, ms2, ms3, ms4 = null
        plt.figure(figsize=(12,8))
        plt.hist(pte_array[count, :])
        plt.savefig(f"{pte_dir}/hist_{mode}_{ms1}x{ms2}-{ms3}x{ms4}.png")
        plt.clf()
        plt.close()
