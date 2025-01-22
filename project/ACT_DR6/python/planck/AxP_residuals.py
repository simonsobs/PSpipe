"""
Plot a suit of residuals for ACT and Planck
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

legacy_dir = "dr6xlegacy/"
npipe_dir = "dr6xnpipe/"

cov_dir_run_a = f"{legacy_dir}/covariances"
spec_dir_run_a = f"{legacy_dir}/spectra_leak_corr_planck_bias_corr"
bf_dir_run_a = f"{legacy_dir}/best_fits"

cov_dir_run_b = f"{npipe_dir}/covariances"
spec_dir_run_b = f"{npipe_dir}/spectra_leak_corr_planck_bias_corr"
bf_dir_run_b = f"{npipe_dir}/best_fits"

run_a_name = "legacy"
run_b_name = "NPIPE"

plot_dir = "paper_plot"
pspy_utils.create_directory(plot_dir)


tested_spectra = ["TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
combination = ["AxA-PxP", "AxP-PxP"]
null_list = {}
# We don't do TT since it's wrong to use the fg model without propagating uncertainties
for spec in tested_spectra:

    null_list[spec, "AxA-PxP"] = []
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f100", "Planck_f100"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f100", "Planck_f100"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f100", "Planck_f100"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f100", "Planck_f100"]]

    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f143", "Planck_f143"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f143", "Planck_f143"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f143", "Planck_f143"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f143", "Planck_f143"]]

    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa5_f090", "dr6_pa5_f090", "Planck_f217", "Planck_f217"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa5_f150", "dr6_pa5_f150", "Planck_f217", "Planck_f217"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa6_f090", "dr6_pa6_f090", "Planck_f217", "Planck_f217"]]
    null_list[spec, "AxA-PxP"] += [[spec, "dr6_pa6_f150", "dr6_pa6_f150", "Planck_f217", "Planck_f217"]]

    null_list[spec, "AxP-PxP"] = []
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa5_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa5_f150", "Planck_f100", "Planck_f100", "Planck_f100"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa6_f090", "Planck_f100", "Planck_f100", "Planck_f100"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa6_f150", "Planck_f100", "Planck_f100", "Planck_f100"]]

    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa5_f090", "Planck_f143", "Planck_f143", "Planck_f143"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa5_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa6_f090", "Planck_f143", "Planck_f143", "Planck_f143"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa6_f150", "Planck_f143", "Planck_f143", "Planck_f143"]]

    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa5_f090", "Planck_f217", "Planck_f217", "Planck_f217"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa5_f150", "Planck_f217", "Planck_f217", "Planck_f217"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa6_f090", "Planck_f217", "Planck_f217", "Planck_f217"]]
    null_list[spec, "AxP-PxP"] += [[spec, "dr6_pa6_f150", "Planck_f217", "Planck_f217", "Planck_f217"]]


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



for spec in tested_spectra:
    for comb in combination:
    

        plt.figure(figsize=(20, 25))

        count = 0
        for null in null_list[spec, comb]:

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

            if "beam_cov" in cov_type_list:
                r_cov_run_a += res_cov_dict_run_a["beam_cov"]
                r_cov_run_b += res_cov_dict_run_b["beam_cov"]

            if "leakage_cov" in cov_type_list:
                r_cov_run_a += res_cov_dict_run_a["leakage_cov"]
                r_cov_run_b += res_cov_dict_run_b["leakage_cov"]

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

            plt.subplot(6, 2, count+1)
            plot_title = f"{ms3}x{ms4} - {ms1}x{ms2}" # we revert the null for plot clarity
            plot_title = plot_title.replace("dr6_", "")
            plot_title = plot_title.replace("-"," vs ")
            plot_title = plot_title.replace("_", "-")

            plt.title(plot_title, fontsize=20)
 
            plt.errorbar(lb - 8, -(res_ps_run_a - res_fg_b) * lb ** l_pow,
                        yerr=np.sqrt(r_cov_run_a.diagonal()) * lb ** l_pow,
                        ls="None", marker = ".",
                        label=f"{run_a_name}   p= {pte_run_a:.3f}", color="lightseagreen")
                     
            plt.errorbar(lb + 8, -(res_ps_run_b - res_fg_b) * lb ** l_pow,
                        yerr=np.sqrt(r_cov_run_b.diagonal()) * lb ** l_pow,
                        ls="None", marker = ".",
                        label=f"{run_b_name}   p= {pte_run_b:.3f}", color="blue", alpha=0.7)

            plt.plot(lb_fg, lb_fg*0, linestyle="--", color="black")
            plt.legend(fontsize=15, loc="lower right")
            plt.axvspan(xmin=0, xmax=xleft,  color="gray", alpha=0.7)
            if xright != lb[-1]:
                plt.axvspan(xmin=xright, xmax=lb[-1], color="gray", alpha=0.7)
        
            plt.xlim(300, 2000)
            if ylims is not None:
                plt.ylim(*ylims)
            plt.xlabel(r"$\ell$", fontsize=25)
    
            if l_pow != 0:
                plt.ylabel(r"$\ell^{%d} \Delta D_\ell^\mathrm{%s}$" % (l_pow, mode), fontsize=25)
            else:
                plt.ylabel(r"$\Delta D_\ell^\mathrm{%s}$" % (mode), fontsize=25)

            print(f"{spec} {ms3}x{ms4} - {ms1}x{ms2}, {run_a_name}   p= {pte_run_a:.3f}")
            print(f"{spec} {ms3}x{ms4} - {ms1}x{ms2}, {run_b_name}   p= {pte_run_b:.3f}")

            count += 1
    
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{spec}_{comb}.pdf", bbox_inches='tight')
        plt.clf()
        plt.close()





