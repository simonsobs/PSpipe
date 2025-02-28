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
import pickle

labelsize = 14

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

remove_first_bin = True

plot_dir = "paper_plot"
pspy_utils.create_directory(plot_dir)

runs = ["legacy", "NPIPE"]
data_dir = {}
data_dir["legacy"] =  "dr6xlegacy/"
data_dir["NPIPE"] =  "dr6xnpipe/"

map_set_list = pspipe_list.get_map_set_list(d)
cov_type_list = ["analytic_cov", "mc_cov", "leakage_cov"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

all_ps, all_cov = {}, {}
for run in runs:
    d_dir = data_dir[run]
    cov_dir = f"{d_dir}/covariances"
    spec_dir= f"{d_dir}/spectra_leak_corr_planck_bias_corr"
    lb, all_ps[run], all_cov[run] = AxP_utils.read_data(map_set_list, spec_dir, cov_dir, cov_type_list, spectra)

tested_spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
combination = ["AxA-PxP", "AxP-PxP"]
null_list = {}
#  TT is wrong because we fix the fg model without propagating uncertainties
# There is evidence that fixing beta_CIB in ACT doesn't fit planck well
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


# Load foreground best fits
fg_file_name = "dr6xlegacy/best_fits" + "/fg_{}x{}.dat"
l_fg, fg_dict = best_fits.fg_dict_from_files(fg_file_name, map_set_list, d["lmax"], spectra=spectra)

multipole_range, l_pows, y_lims = AxP_utils.get_plot_params()


pte_dict = {}

for spec in tested_spectra:
    for comb in combination:
        pte_dict[spec, comb, "legacy"] = {}
        pte_dict[spec, comb, "NPIPE"] = {}

    
        plt.figure(figsize=(20, 25))

        count = 0
        for null in null_list[spec, comb]:

            mode, ms1, ms2, ms3, ms4 = null
            
            res_fg = fg_dict[ms1, ms2][mode] - fg_dict[ms3, ms4][mode]
            lb_fg, res_fg_b = pspy_utils.naive_binning(l_fg, res_fg, d["binning_file"], d["lmax"])
            if remove_first_bin:
                lb_fg, res_fg_b = lb_fg[1:], res_fg_b[1:]

            plt.subplot(6, 2, count+1)
            plot_title = f"{ms3}x{ms4} - {ms1}x{ms2}" # we revert the null for plot clarity
            plot_title = plot_title.replace("dr6_", "")
            plot_title = plot_title.replace("-"," vs ")
            plot_title = plot_title.replace("_", " ")
            plot_title = plot_title.replace("pa", "PA")

            plt.title(plot_title, fontsize=20)
            plt.plot(lb_fg, lb_fg * 0, linestyle="--", color="black")

            lmin, lmax = AxP_utils.get_lmin_lmax(null, multipole_range)
            lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
            l_pow = l_pows[mode]
            ylims = y_lims[mode]
            xleft, xright = lb[lrange][0], lb[lrange][-1]

            colors = ["lightseagreen", "blue"]

            for r_count, (run, col) in enumerate(zip(runs, colors)):
                res_cov_dict = {}
            
                for cov in cov_type_list:
                    lb, res_ps, res_cov_dict[cov] = consistency.compare_spectra([ms1, ms2, ms3, ms4],
                                                                                "ab-cd",
                                                                                all_ps[run],
                                                                                all_cov[run][cov],
                                                                                mode = mode,
                                                                                return_chi2=False)
                                                                    
                    if remove_first_bin:
                        lb, res_cov_dict[cov]= lb[1:], res_cov_dict[cov][1:,1:]
                        res_ps = res_ps[1:]

                name = "analytical"
                if "mc_cov" in cov_type_list:
                    r_cov = covariance.correct_analytical_cov(res_cov_dict["analytic_cov"],
                                                              res_cov_dict["mc_cov"],
                                                              only_diag_corrections=True)
                                                  
                if "beam_cov" in cov_type_list:
                    r_cov += res_cov_dict["beam_cov"]
    
                if "leakage_cov" in cov_type_list:
                    r_cov += res_cov_dict["leakage_cov"]
                    
                chi2 = (res_ps[lrange] - res_fg_b[lrange]) @ np.linalg.inv(r_cov[np.ix_(lrange, lrange)]) @ (res_ps[lrange] - res_fg_b[lrange])

                ndof = len(lb[lrange]) - 1
                pte = 1 - ss.chi2(ndof).cdf(chi2)
                
                print(f"{spec} {plot_title}, {run}   p= {100*pte:.0f} %")

                
                pte_dict[spec, comb, run][f"{ms1}x{ms2} - {ms3}x{ms4}"] = pte

                

                plt.errorbar(lb - 8 + r_count*16, -(res_ps - res_fg_b) * lb ** l_pow,
                            yerr=np.sqrt(r_cov.diagonal()) * lb ** l_pow,
                            ls="None", marker = ".",
                            label=f"{run}   p= {100*pte:.0f} %", color=col)
                     
            plt.legend(fontsize=15, loc="lower right")
            plt.xlabel(r"$\ell$", fontsize=25)
            plt.xlim(300, 2000)
            plt.axvspan(xmin=0, xmax=xleft,  color="gray", alpha=0.7)
            
            if xright != lb[-1]:
                plt.axvspan(xmin=xright, xmax=lb[-1], color="gray", alpha=0.7)
                    
            if ylims is not None:
                plt.ylim(*ylims)
                
            if l_pow != 0:
                plt.ylabel(r"$\ell^{%d} \Delta D_\ell^\mathrm{%s} \ [\mu K^{2}]$" % (l_pow, mode), fontsize=25)
            else:
                plt.ylabel(r"$\Delta D_\ell^\mathrm{%s} \ [\mu K^{2}] $" % (mode), fontsize=25)


            plt.tick_params(labelsize=labelsize)

            count += 1
            
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{spec}_{comb}.pdf", bbox_inches='tight')
        plt.clf()
        plt.close()


pickle.dump(pte_dict, open(f"{plot_dir}/pte_dict.pkl", "wb"))


