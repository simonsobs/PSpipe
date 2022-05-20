"""
This script plot and compare the covariance matrix elements
"""

import matplotlib
matplotlib.use("Agg")
import pylab as plt
import numpy as np
from pspy import so_spectra, so_cov, so_mcm, pspy_utils, so_map, so_dict
import scipy.interpolate
import sys
import SO_noise_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


scan_list = d["scan_list"]
lmax = d["lmax"]
niter = d["niter"]
spectra = d["spectra"]
split_list = d["split_list"]
runs = d["runs"]
spin_pairs = d["spin_pairs"]
binning_file = d["binning_file_name"]
clfile = d["clfile"]

lth, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)

spectra_dir = "spectra"
plot_dir = "plot/covariance"
cov_dir = "covariance"
window_dir = "windows"


pspy_utils.create_directory(plot_dir)



# Check effect of weighting
for bl in ["TTTT", "TETE", "EEEE"]:
    for scan in scan_list:
        for run in runs:
            _name = "%s_%sx%s_%s" % (scan, "split0", "split0", run)
            lb, _ = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, _name), spectra=spectra)
            n_bins = len(lb)
            
            cov = np.load("%s/analytic_cov_%s_%s.npy" % (cov_dir, scan, run))
            cov_select = so_cov.selectblock(cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)

            var = cov_select.diagonal()
            
            plt.semilogy()
            plt.plot(lb, np.sqrt(var), label = run)
            
        plt.title(r"$\sigma^{%s}_{\ell}$, %s" % (bl, scan), fontsize=14)
        plt.xlabel(r"$\ell$", fontsize=14)

        plt.legend()

        plt.savefig("%s/sigma_%s_%s.png" % (plot_dir, bl, scan), bbox_inches="tight")
        plt.clf()
        plt.close()


# Check effect of approximating
for bl in ["TTTT", "TETE", "EEEE"]:
    for scan in scan_list:
        for run in runs:
            _name = "%s_%sx%s_%s" % (scan, "split0", "split0", run)
            lb, _ = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, _name), spectra=spectra)
            n_bins = len(lb)
            
            cov = np.load("%s/analytic_cov_%s_%s.npy" % (cov_dir, scan, run))
            quick_cov = np.load("%s/quick_cov_%s_%s.npy" % (cov_dir, scan, run))

            cov_select = so_cov.selectblock(cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            quick_cov_select = so_cov.selectblock(quick_cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            
            var = cov_select.diagonal()
            quick_var = quick_cov_select.diagonal()
            
            plt.semilogy()

            plt.plot(lb, np.sqrt(var), ".", label = run, color = "blue")
            plt.plot(lb, np.sqrt(quick_var), "-", label = "approx %s" % run, color = "blue")

            plt.title(r"$\sigma^{%s}_{\ell}$, %s" % (bl, scan), fontsize=14)
            plt.xlabel(r"$\ell$", fontsize=14)

            plt.legend()
            plt.savefig("%s/test_approx_sigma_%s_%s_%s.png" % (plot_dir, bl, scan, run), bbox_inches="tight")
            plt.clf()
            plt.close()

colors = ["blue", "green", "orange", "red", "grey", "lightblue", "magenta"]


for bl in ["TTTT", "TETE", "EEEE"]:
    plt.figure(figsize = (15, 10))

    for c_scan, scan in enumerate(scan_list):
        fmt = ["-", "--"]

        for c_run, run in enumerate(runs):
    
            _name = "%s_%sx%s_%s" % (scan, "split0", "split0", run)
            lb, _ = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, _name), spectra=spectra)
            n_bins = len(lb)
        
            cov = np.load("%s/analytic_cov_%s_%s.npy" % (cov_dir, scan, run))
            cov_select = so_cov.selectblock(cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)

            var = cov_select.diagonal()
            
            plt.semilogy()
            plt.plot(lb, np.sqrt(var), fmt[c_run], label = "%s %s" % (scan, run), color = colors[c_scan])
    
    
    plt.title(r"$\sigma^{%s}_{\ell}$" % (bl), fontsize=14)
    plt.xlabel(r"$\ell$", fontsize=14)

    plt.legend()
    plt.savefig("%s/sigma_%s.png" % (plot_dir, bl), bbox_inches="tight")
    plt.clf()
    plt.close()


