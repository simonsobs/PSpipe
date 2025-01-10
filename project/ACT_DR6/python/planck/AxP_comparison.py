"""
Compare ACT DR6 spectra with Planck NPIPE and legacy spectra
"""
from pspy import so_dict, pspy_utils, so_cov
from pspipe_utils import consistency, best_fits, covariance, pspipe_list
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
import AxP_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
remove_first_bin = True
part = "part_b"


legacy_dir = "dr6xlegacy/"
npipe_dir = "dr6xnpipe/"
bestfit_dir = "best_fits"
spec_dir = "spectra_leak_corr_planck_bias_corr"
cov_dir = "covariances"

cov_dir_run_a = f"{legacy_dir}/{cov_dir}"
cov_dir_run_b = f"{npipe_dir}/{cov_dir}"

spec_dir_run_a = f"{legacy_dir}/{spec_dir}"
spec_dir_run_b = f"{npipe_dir}/{spec_dir}"

bf_dir_run_a = f"{legacy_dir}/{bestfit_dir}"
bf_dir_run_b = f"{npipe_dir}/{bestfit_dir}"

run_a_name = "legacy"
run_b_name = "npipe"

plot_dir = "AxP_plots"
pspy_utils.create_directory(f"{plot_dir}/{part}")



null_list = {}
null_list["part_a"] = []
null_list["part_a"] += [["EE", "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f143", "Planck_f143"]]
null_list["part_a"] += [["EE", "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f100", "Planck_f100"]]
null_list["part_a"] += [["ET", "dr6_pa6_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["part_a"] += [["ET", "dr6_pa5_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]

null_list["part_b"] = []
null_list["part_b"] += [["EE", "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f100", "Planck_f100"]]
null_list["part_b"] += [["EE", "dr6_pa6_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]
null_list["part_b"] += [["EE", "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f100", "Planck_f100"]]
null_list["part_b"] += [["EE", "dr6_pa5_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]
null_list["part_b"] += [["EE", "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["EE", "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["EE", "dr6_pa6_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["EE", "dr6_pa5_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["EE", "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f217", "Planck_f217"]]
null_list["part_b"] += [["EE", "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f217", "Planck_f217"]]

null_list["part_b"] += [["ET", "dr6_pa6_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]
null_list["part_b"] += [["ET", "dr6_pa5_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]

null_list["part_b"] += [["ET", "dr6_pa5_f090", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["ET", "dr6_pa6_f090", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["ET", "dr6_pa6_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["ET", "dr6_pa5_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["TE", "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["TE", "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["TE", "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["TE", "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f143", "Planck_f143"]]
null_list["part_b"] += [["TE", "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f217", "Planck_f217"]]
null_list["part_b"] += [["TE", "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f217", "Planck_f217"]]



cov_type_list = ["analytic_cov", "mc_cov", "leakage_cov"]

map_set_list = pspipe_list.get_map_set_list(d)

lb, all_ps_run_a, all_cov_run_a = AxP_utils.read_data(map_set_list, spec_dir_run_a, cov_dir_run_a, cov_type_list, spectra)
lb, all_ps_run_b, all_cov_run_b = AxP_utils.read_data(map_set_list, spec_dir_run_b, cov_dir_run_b, cov_type_list, spectra)

# Load foreground best fits
fg_file_name = bf_dir_run_a + "/fg_{}x{}.dat"
l_fg, fg_dict = best_fits.fg_dict_from_files(fg_file_name, map_set_list, d["lmax"], spectra=spectra)

operations = {"diff": "ab-cd"}

multipole_range, l_pows, y_lims = AxP_utils.get_plot_params()
y_lims_spec = {}
y_lims_spec["TT"] = [0, 2*10**9]
y_lims_spec["TE"] = [-150, 150]
y_lims_spec["ET"] = [-150, 150]
y_lims_spec["EE"] = [-10, 70]


print(f"we will do {len(null_list)} null tests")

for null in null_list[part]:

    print(null)

    mode, ms1, ms2, ms3, ms4 = null
    res_cov_dict_run_a, res_cov_dict_run_b = {}, {}
    
    l, Db1_run_a, sigma1_run_a =  AxP_utils.read_ps_and_sigma(spec_dir_run_a, cov_dir_run_a, ms1, ms2, mode, cov_type_list)
    l, Db2_run_a, sigma2_run_a =  AxP_utils.read_ps_and_sigma(spec_dir_run_a, cov_dir_run_a, ms3, ms4, mode, cov_type_list)
    
    l, Db1_run_b, sigma1_run_b =  AxP_utils.read_ps_and_sigma(spec_dir_run_b, cov_dir_run_b, ms1, ms2, mode, cov_type_list)
    l, Db2_run_b, sigma2_run_b =  AxP_utils.read_ps_and_sigma(spec_dir_run_b, cov_dir_run_b, ms3, ms4, mode, cov_type_list)

    if remove_first_bin:
        Db1_run_a, sigma1_run_a, Db2_run_a, sigma2_run_a = Db1_run_a[1:], sigma1_run_a[1:], Db2_run_a[1:], sigma2_run_a[1:]
        Db1_run_b, sigma1_run_b, Db2_run_b, sigma2_run_b = Db1_run_b[1:], sigma1_run_b[1:], Db2_run_b[1:], sigma2_run_b[1:]
        
    for cov in cov_type_list:
        lb, res_ps_run_a, res_cov_dict_run_a[cov] = consistency.compare_spectra([ms1, ms2, ms3, ms4],
                                                                                "ab-cd",
                                                                                all_ps_run_a,
                                                                                all_cov_run_a[cov],
                                                                                mode = mode,
                                                                                return_chi2=False)
                                                                    
        lb, res_ps_run_b, res_cov_dict_run_b[cov] = consistency.compare_spectra([ms1, ms2, ms3, ms4],
                                                                                "ab-cd",
                                                                                all_ps_run_b,
                                                                                all_cov_run_b[cov],
                                                                                mode = mode,
                                                                                return_chi2=False)

        if remove_first_bin:
            lb, res_cov_dict_run_a[cov] = lb[1:], res_cov_dict_run_a[cov][1:,1:]
            res_ps_run_a = res_ps_run_a[1:]
            res_cov_dict_run_b[cov] = res_cov_dict_run_b[cov][1:,1:]
            res_ps_run_b = res_ps_run_b[1:]


    res_fg = fg_dict[ms1, ms2][mode] - fg_dict[ms3, ms4][mode]
    lb_fg, res_fg_b = pspy_utils.naive_binning(l_fg, res_fg, d["binning_file"], d["lmax"])
    
    if remove_first_bin:
        lb_fg, res_fg_b = lb_fg[1:], res_fg_b[1:]

    
    name = "analytical"
    if "mc_cov" in cov_type_list:
        r_cov_run_a = covariance.correct_analytical_cov(res_cov_dict_run_a["analytic_cov"],
                                                         res_cov_dict_run_a["mc_cov"],
                                                         only_diag_corrections=True)
                                                  
        r_cov_run_b = covariance.correct_analytical_cov(res_cov_dict_run_b["analytic_cov"],
                                                        res_cov_dict_run_b["mc_cov"],
                                                        only_diag_corrections=True)

        name += "+mc_corr"
    if "beam_cov" in cov_type_list:
        r_cov_run_a += res_cov_dict_run_a["beam_cov"]
        r_cov_run_b += res_cov_dict_run_b["beam_cov"]

        name += "+beam"

    if "leakage_cov" in cov_type_list:
        r_cov_run_a += res_cov_dict_run_a["leakage_cov"]
        r_cov_run_b += res_cov_dict_run_b["leakage_cov"]

        name += "+leakage"


    corr_run_a = so_cov.cov2corr(r_cov_run_a)
    

    
    lmin, lmax = AxP_utils.get_lmin_lmax(null, multipole_range)

    # Plot residual and get chi2
    lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
    fname = f"diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}"
    l_pow = l_pows[mode]
    
    expected_res = 0.
    ylims = y_lims[mode]
    if "pa4" in fname:
        ylims = (y_lims[mode][0] * 5,  y_lims[mode][1] * 5)

                                     
    chi2_run_a = (res_ps_run_a[lrange] - res_fg_b[lrange]) @ np.linalg.inv(r_cov_run_a[np.ix_(lrange, lrange)]) @ (res_ps_run_a[lrange] - res_fg_b[lrange])
    chi2_run_b = (res_ps_run_b[lrange] - res_fg_b[lrange]) @ np.linalg.inv(r_cov_run_b[np.ix_(lrange, lrange)]) @ (res_ps_run_b[lrange] - res_fg_b[lrange])

    ndof = len(lb[lrange])
    ndof -= 1

    pte_run_a = 1 - ss.chi2(ndof).cdf(chi2_run_a)
    pte_run_b = 1 - ss.chi2(ndof).cdf(chi2_run_b)

    xleft, xright = lb[lrange][0], lb[lrange][-1]
    
    if mode == "TT":
        l_pow_plain = 2
    else:
        l_pow_plain = 0

    plt.figure(figsize=(16, 10))
    plot_title = f"{ms3}x{ms4} - {ms1}x{ms2}"
    plot_title = plot_title.replace("dr6_", "")
    plt.suptitle(plot_title.replace("-"," vs "))
    plt.subplot(2,1,1)
    plt.errorbar(lb - 2, Db1_run_a * lb ** l_pow_plain, sigma1_run_a * lb ** l_pow_plain, label=f"{ms1}x{ms2} [{run_a_name} run]", color="grey", fmt=".", markersize=0)
    plt.errorbar(lb + 2, Db1_run_b * lb ** l_pow_plain, sigma1_run_b * lb ** l_pow_plain, label=f"{ms1}x{ms2} [{run_b_name} run]", color="black", fmt=".", markersize=0)

    plt.errorbar(lb - 8, (Db2_run_a + res_fg_b) * lb ** l_pow_plain, sigma2_run_a * lb ** l_pow_plain, label=f"{ms3}x{ms4} + expected fg diff [{run_a_name} run]", fmt=".", color="royalblue", markersize=0)
    
    plt.errorbar(lb + 8, (Db2_run_b + res_fg_b) * lb ** l_pow_plain, sigma2_run_b * lb ** l_pow_plain, label=f"{ms3}x{ms4} + expected fg diff [{run_b_name} run]", fmt=".", color="darkorange", markersize=0)
    plt.legend()
    plt.xlabel(r"$\ell$", fontsize=18)
    plt.ylabel(r"$\ell^{%d} D_\ell^\mathrm{%s}$" % (l_pow_plain, mode), fontsize=18)

    plt.ylim(y_lims_spec[mode][0], y_lims_spec[mode][1])
    plt.xlim(300, 2200)

    plt.axvspan(xmin=0, xmax=xleft,  color="gray", alpha=0.7)
    if xright != lb[-1]:
        plt.axvspan(xmin=xright, xmax=lb[-1], color="gray", alpha=0.7)

    plt.subplot(2,1,2)
    plt.errorbar(lb - 8, -(res_ps_run_a - res_fg_b) * lb ** l_pow,
                     yerr=np.sqrt(r_cov_run_a.diagonal()) * lb ** l_pow,
                     ls="None", marker = ".",
                     label=f"[{run_a_name}  run] {name} [$\chi^2 = {{{chi2_run_a:.1f}}}/{{{ndof}}}$ (${{{pte_run_a:.4f}}}$)]", color="royalblue", alpha=0.7)
                     
    plt.errorbar(lb + 8, -(res_ps_run_b - res_fg_b) * lb ** l_pow,
                yerr=np.sqrt(r_cov_run_b.diagonal()) * lb ** l_pow,
                ls="None", marker = ".",
                label=f"[{run_b_name} run]  {name} [$\chi^2 = {{{chi2_run_b:.1f}}}/{{{ndof}}}$ (${{{pte_run_b:.4f}}}$)]", color="darkorange", alpha=0.7)

    plt.plot(lb_fg, lb_fg*0, color="black")
    plt.legend()
    plt.axvspan(xmin=0, xmax=xleft,  color="gray", alpha=0.7)
    if xright != lb[-1]:
        plt.axvspan(xmin=xright, xmax=lb[-1], color="gray", alpha=0.7)
        
    plt.title(plot_title.replace("dr6_", ""))
    plt.xlim(300, 2200)
    if ylims is not None:
        plt.ylim(*ylims)
    plt.xlabel(r"$\ell$", fontsize=18)
    plt.ylabel(r"$\ell^{%d} \Delta D_\ell^\mathrm{%s}$" % (l_pow, mode), fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{part}/{fname}.png", bbox_inches='tight')
    plt.clf()
    plt.close()

