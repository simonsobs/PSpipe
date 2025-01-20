"""
This script test the bias and covariance of combined spectra
"""

from pspy import so_dict, pspy_utils, so_spectra
from pspipe_utils import log
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as stats

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


bestfit_dir = f"best_fits"
combined_spec_dir = f"combined_sim_spectra_syst"
plot_dir = f"plots/combined_sim_spectra_syst/"

pspy_utils.create_directory(plot_dir)

binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
iStart = d["iStart"]
iStop =  d["iStop"]

name = "all"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
my_spectra = ["TT", "TE", "TB", "EE", "EB", "BB"]
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

lth, Dlth = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

for spectrum in my_spectra:
    print(spectrum)

    lb, Dbth = pspy_utils.naive_binning(lth, Dlth[spectrum], binning_file, lmax)

    cov_ml = np.load(f"{combined_spec_dir}/cov_{name}_{spectrum}.npy")
    
  #  Dl_list = []
    Dl_list_fg_sub = []

    inv_cov_ml = np.linalg.inv(cov_ml)

    chi2_list = []
    for iii in range(iStart, iStop + 1):
       # l, Dl, sigma = np.loadtxt(f"{combined_spec_dir}/Dl_{name}_{spectrum}_{iii:05d}.dat", unpack=True)
        l, Dl_fg_sub, sigma = np.loadtxt(f"{combined_spec_dir}/Dl_{name}_{spectrum}_{iii:05d}_cmb_only.dat", unpack=True)
       # Dl_list += [Dl]
        Dl_list_fg_sub += [Dl_fg_sub]
        
        id = np.where(lb >= l[0])

        chi2 = (Dl_fg_sub - Dbth[id]) @  inv_cov_ml @  (Dl_fg_sub - Dbth[id])
        
        chi2_list += [chi2]


    n_dof = len(Dl_fg_sub)
    
    plt.hist(chi2_list, bins=40, density=True, histtype="step", label="sims chi2 distribution")
    x = np.arange(n_dof - 60, n_dof + 60)
    plt.plot(
        x,
        stats.chi2.pdf(x, df=n_dof),
        "-",
        linewidth=2,
        color="orange",
        label="expected distribution",
    )
    plt.savefig(f"{plot_dir}/histogram_{spectrum}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
    
  #  Dl_mean = np.mean(Dl_list, axis=0)
    Dl_fg_sub_mean = np.mean(Dl_list_fg_sub, axis=0)

    print("ok")
    cov = np.cov(Dl_list_fg_sub, rowvar=False)
    
    sigma_mc = np.sqrt(cov.diagonal())
    sigma_analytic = np.sqrt(cov_ml.diagonal())
    
    plt.figure(figsize=(12,8))
    plt.xlabel(r"$\ell$", fontsize=19)
    plt.ylabel(r"$D^{%s}_{\ell}$" % spectrum, fontsize=19)
    plt.plot(lb, Dbth)
  #  plt.errorbar(l, Dl_mean, sigma_mc, fmt=".", label="mean spectrum")
    plt.errorbar(l, Dl_fg_sub_mean, sigma_mc, fmt=".", label="mean spectrum fg subtracted")
    plt.legend()
    plt.savefig(f"{plot_dir}/{spectrum}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.xlabel(r"$\ell$", fontsize=19)
    plt.ylabel(r"$D^{%s}_{\ell} - D^{%s, th}_{\ell}$" % (spectrum,spectrum), fontsize=19)
 #   plt.errorbar(l, Dl_mean - Dbth[id], sigma_mc / np.sqrt(iStop + 1 - iStart), label="mean - theory")
    plt.errorbar(l, Dl_fg_sub_mean - Dbth[id], sigma_mc / np.sqrt(iStop + 1 - iStart), label="mean fg subtracted - theory")
    plt.legend()
    plt.savefig(f"{plot_dir}/{spectrum}_residual.png", bbox_inches="tight")
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.semilogy()
    plt.xlabel(r"$\ell$", fontsize=19)
    plt.ylabel(r"$\sigma^{%s}_{\ell}$" % (spectrum), fontsize=19)
    plt.errorbar(l, sigma_mc, label="covariance of the ML combination of the simulations")
    plt.errorbar(l, sigma_analytic, label="expected ML combination covariance")
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel(r"$\ell$", fontsize=19)
    plt.ylabel(r"$\sigma^{%s, mc}_{\ell}/\sigma^{%s, max likelihood}_{\ell}$" % (spectrum, spectrum), fontsize=19)
    plt.errorbar(l, sigma_mc/sigma_analytic, label="mc errors/max likelihood errors")
    plt.legend()
    plt.savefig(f"{plot_dir}/error_{spectrum}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
