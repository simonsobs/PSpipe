"""
This script plot the ratio of power spectra of ACT/SPT3G/Planck with respect to  Planck best fit.
we also fit a straght line to the ratio
"""

from pspy import so_dict, so_spectra, pspy_utils
from pspipe_utils import  log, best_fits, external_data, covariance
import numpy as np
import pylab as plt
import sys, os
from matplotlib import rcParams
import pspipe_utils
from scipy import optimize



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

def straight_line(ell, a, b):
    return a * ell + b

def get_slope_and_error(ell, ratio, sigma_ratio, use_scipy=False):

    if use_scipy == True:
        pars, pars_covariance = optimize.curve_fit(straight_line, ell, ratio, sigma=sigma_ratio)
        a, sigma_a = pars[0], np.sqrt(pars_covariance[0,0] )
        b, sigma_b = pars[1], np.sqrt(pars_covariance[1,1] )
    else:
        n_obs = len(ell)
        P = np.zeros((n_obs, 2))
        for pow in range(2):
            P[:, pow] = (ell) ** pow

        inv_cov = np.diag(1 / sigma_ratio ** 2)
        param_cov = np.linalg.inv(P.T @ inv_cov @ P)
        param_estimated = param_cov @ P.T @ inv_cov @ ratio

        a, sigma_a = param_estimated[1], np.sqrt(param_cov[1,1] )
        b, sigma_b = param_estimated[0], np.sqrt(param_cov[0,0] )

    return a, b, sigma_a, sigma_b


rcParams["font.family"] = "serif"
rcParams["font.size"] = "12"
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
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

use_scipy = False
add_str = ""
if use_scipy:
    add_str = "scipy"

spt3G_ratio_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/spt3G")
l_spt_EE, frac_spt_EE, sigma_spt_EE = np.loadtxt(f"{spt3G_ratio_path}/spt-3g_2024_Dl_EE_delensed_unblinded_theoryrel_digitized.txt", unpack=True)

plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(plot_dir)

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")
binning_planck = f"{planck_data_path}/bin_planck.dat"

    
combined_spec_dir = "combined_spectra_Planck"
bestfit_dir = "best_fits_Planck"
lth, Dl_th = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

# First do ACT
lb_th, Db_th = pspy_utils.naive_binning(lth, Dl_th["EE"], binning_file, lmax)
lb_ml, vec_ml, sigma_ml = np.loadtxt(f"{combined_spec_dir}/{type}_all_EE_cmb_only.dat", unpack=True)
id = np.where(lb_th >= lb_ml[0])
Db_th = Db_th[id]
ratio_ACT, sigma_ratio_ACT = vec_ml / Db_th, sigma_ml / Db_th

# Now do Planck, it is given in cl
        
l_planck, ps_planck_b, sigma_planck, cov_planck = external_data.get_planck_cmb_only_data()

cl_th = Dl_th["EE"] * 2 * np.pi / (lth * (lth + 1))
lb_th_p, cb_th_p = pspy_utils.naive_binning(lth, cl_th, binning_planck, lmax)
l_planck_spec = l_planck["EE"]

id_planck = np.where( (lb_th_p >= l_planck_spec[0]) & (lb_th_p <= l_planck_spec[-1]))
cb_th_p = cb_th_p[id_planck]
ratio_Planck, sigma_ratio_Planck = ps_planck_b["EE"]/cb_th_p, sigma_planck["EE"]/cb_th_p
        
rebin_fac = 20

# in that case we rebin using the cov mat
cov = cov_planck["EEEE"]
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

plt.figure(figsize=(12,8))

plt.plot(lth, lth * 0 + 1, color="black", linestyle="--")
plt.errorbar(lb_ml, ratio_ACT, sigma_ratio_ACT, fmt="o", label="ACT", color="blue")
plt.errorbar(l_planck_spec, ratio_Planck, sigma_ratio_Planck, fmt="o", label="Planck", color="darkorange")
plt.errorbar(l_spt_EE, frac_spt_EE, sigma_spt_EE, fmt="o", label="SPT3G (pre-unblinding)", color="red")

id = np.where(l_planck_spec > 400)
lmin_round = int(np.floor(l_planck_spec[id][0] / 100.0)) * 100


a_ACT, b_ACT, sigma_a_ACT, sigma_b_ACT = get_slope_and_error(lb_ml, ratio_ACT, sigma_ratio_ACT, use_scipy)
a_Planck, b_Planck, sigma_a_Planck, sigma_b_Planck = get_slope_and_error(l_planck_spec[id], ratio_Planck[id], sigma_ratio_Planck[id], use_scipy)
a_SPT, b_SPT, sigma_a_SPT, sigma_b_SPT = get_slope_and_error(l_spt_EE, frac_spt_EE, sigma_spt_EE, use_scipy)

plt.title(r" $ (a \times 10^{-5}) \ell + b$", fontsize=30)
plt.plot(lb_ml, straight_line(lb_ml, a_ACT, b_ACT), "--", label=r"$a_{\rm ACT} = %.2f \pm %.2f  $" % (a_ACT * 10 ** 5, sigma_a_ACT * 10 ** 5) , color="blue")
plt.plot(l_planck_spec[id], straight_line(l_planck_spec[id], a_Planck, b_Planck), "--", label=r"$a_{\rm Planck} = %.2f \pm %.2f [\ell_{\rm min} = %d ] $" % (a_Planck * 10 ** 5, sigma_a_Planck * 10 ** 5, lmin_round) , color="darkorange")
plt.plot(l_spt_EE, straight_line(l_spt_EE, a_SPT, b_SPT), "--", label=r"$a_{\rm SPT} = %.2f \pm %.2f$ " % (a_SPT * 10 ** 5, sigma_a_SPT * 10 ** 5) , color="red")
plt.ylim(0.9, 1.2)
plt.xlim(30, 3000)
plt.legend(fontsize=18)
plt.ylabel(r"$D^{\rm data}_\ell$/$D^{\rm th}_{\ell, \rm Planck} $", fontsize=30)

plt.xlabel(r"$\ell$", fontsize=30)
plt.yticks(np.linspace(0.90, 1.2, 7)[1:-1])
plt.savefig(f"{plot_dir}/ratio_EE_with_SPT{extra_str}{add_str}.pdf", bbox_inches="tight")
plt.clf()
plt.close()
