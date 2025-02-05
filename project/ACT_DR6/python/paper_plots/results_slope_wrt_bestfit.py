"""
This script plot the ratio of power spectra of ACT/Planck with respect to ACT/P-ACT and Planck best fit.
Before running it you need to have run: get_best_fit_mflike, apply_likelihood_calibration, get_combined_spectra
with post_likelihood, post_likelihood_Planck, post_likelihood_PACT

"""

from pspy import so_dict, so_spectra, pspy_utils
from pspipe_utils import  log, best_fits, external_data, covariance
import numpy as np
import pylab as plt
import sys, os
from matplotlib import rcParams
import pspipe_utils
import scipy.stats as ss

def rebin_spectrum_with_diag_cov(l, ps, cov, rebin_fac=5):

    nbin = len(l)
    new_nbin = int(np.ceil(nbin / rebin_fac))
    P_mat = np.zeros((nbin, new_nbin))

    for i in range(nbin):
        ii = i // rebin_fac
        P_mat[i, ii] = 1
    
    W = np.diag(1 / np.diag(cov))
    cov_binned = np.linalg.inv(P_mat.T @ W @ cov @ W @ P_mat)
    vec_binned = np.linalg.inv(P_mat.T @ W @ P_mat) @  P_mat.T @ W @ ps
    
    # assume a non weighted average for multipoles
    lb_binned = np.linalg.inv(P_mat.T @ P_mat) @ P_mat.T @ l
                                 
    return lb_binned, vec_binned, cov_binned


rcParams["font.family"] = "serif"
rcParams["font.size"] = "12"
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 12

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]
binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
use_diag_weighting = True
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
########################################################################################
spec_list = ["TT", "EE"]
########################################################################################


run_name = {}
run_name["_paper"] = "ACT"
run_name["_paper_PACT"] = "PACT"
run_name["_Planck"] = "Planck"

plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(plot_dir)

my_tag_list = ["_Planck", "_paper", "_paper_PACT"]
name_list = ["Planck", "ACT", "P-ACT"]

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")
binning_planck = f"{planck_data_path}/bin_planck.dat"


y_ticks={}
y_ticks["TT"] = np.linspace(0.95, 1.1, 7)
y_ticks["EE"] = np.linspace(0.90, 1.2, 7)
print(y_ticks["TT"])

for spec in spec_list:
    plt.figure(figsize=(12,6))
    for count, my_tag in enumerate(my_tag_list):
    
        combined_spec_dir = f"combined_spectra{my_tag}"
        bestfit_dir = f"best_fits{my_tag}"
        lth, Dl_th = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

        # First do ACT
        lb_th, Db_th = pspy_utils.naive_binning(lth, Dl_th[spec], binning_file, lmax)
        lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_all_{spec}_cmb_only.dat", unpack=True)
        id = np.where(lb_th >= lb_ml[0])
        Db_th = Db_th[id]
        ratio_ACT, sigma_ratio_ACT = vec_ml / Db_th, sigma_ml / Db_th


        # Now do Planck, it is given in cl
        
        l_planck, ps_planck_b, sigma_planck, cov_planck = external_data.get_planck_cmb_only_data()

        cl_th = Dl_th[spec] * 2 * np.pi / (lth * (lth + 1))
        lb_th_p, cb_th_p = pspy_utils.naive_binning(lth, cl_th, binning_planck, lmax)
        l_planck_spec = l_planck[spec]
        
        id_planck = np.where( (lb_th_p >= l_planck_spec[0]) & (lb_th_p <= l_planck_spec[-1]))
        cb_th_p = cb_th_p[id_planck]
        ratio_Planck, sigma_ratio_Planck = ps_planck_b[spec]/cb_th_p, sigma_planck[spec]/cb_th_p
        
        if spec == "EE": rebin_fac=20
        if spec == "TT": rebin_fac=8

        # in that case we rebin using the cov mat
        cov = cov_planck[spec+spec]
        cov_ratio = cov / np.outer(cb_th_p, cb_th_p)
        
        
        if use_diag_weighting:
            extra_str= "_diag_weight"
            l_planck_spec, ratio_Planck, cov_ratio_ML = rebin_spectrum_with_diag_cov(l_planck_spec,
                                                                                     ratio_Planck,
                                                                                     cov_ratio,
                                                                                     rebin_fac=rebin_fac)
        else:
            extra_str= "_opt_weight"
            l_planck_spec, ratio_Planck, cov_ratio_ML = covariance.rebin_spectrum_with_cov(l_planck_spec,
                                                                                           ratio_Planck,
                                                                                           cov_ratio,
                                                                                           rebin_fac=rebin_fac)

        sigma_ratio_Planck = np.sqrt(np.diagonal(cov_ratio_ML))
        
        plt.subplot(3, 1, count+1)
        plt.ylabel(r"$D^{\rm data}_\ell$/$D^{\rm th}_{\ell, \rm %s} $" % name_list[count], fontsize=16)
        plt.plot(lth, lth * 0 + 1, color="black", linestyle="--")
        plt.errorbar(lb_ml, ratio_ACT, sigma_ratio_ACT, fmt="-o", label="ACT")
        plt.errorbar(l_planck_spec, ratio_Planck, sigma_ratio_Planck, fmt="-o", label="Planck", alpha=0.7)

        plt.ylim(y_ticks[spec][0], y_ticks[spec][-1])
            
        plt.xlim(30, 4000)
        
        if count == 0:
            plt.legend(loc="upper left",bbox_to_anchor=(0.83, 1.62), fontsize=16, frameon=False)
        if count == 2:
            plt.xlabel(r"$\ell$", fontsize=22)
        if count != 2:
            plt.xticks([])
        plt.yticks(y_ticks[spec][1:-1])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle("Ratio (data / best fit) %s" % spec, fontsize=20, x=0.4)

    plt.savefig(f"{plot_dir}/ratio_{spec}{extra_str}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()
