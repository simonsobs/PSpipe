"""
This script use the covariance matrix block to form a cross array covariance matrix
with block TT - TE - ET - EE
Note that for the ET block, we do not include any same array, same survey spectra, since for
these guys TE = ET
"""

import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils, so_cov
from pspipe_utils import covariance, pspipe_list
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

use_mc_corrected_cov = True
only_diag_corrections = False
use_beam_covariance = d["use_beam_covariance"]

cov_name = "analytic_cov"
if use_mc_corrected_cov:
    if only_diag_corrections:
        cov_name += "_with_diag_mc_corrections"
    else:
        cov_name += "_with_mc_corrections"
if use_beam_covariance:
    cov_name += "_with_beam"

cov_dir = "covariances"
like_product_dir = "like_product"
plot_dir = "plots/combined_cov"


pspy_utils.create_directory(like_product_dir)
pspy_utils.create_directory(plot_dir)

spec_name_list = pspipe_list.get_spec_name_list(d, char="_")

analytic_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                            cov_dir,
                                                            cov_name,
                                                            spectra_order=["TT", "TE", "ET", "EE"],
                                                            remove_doublon=True,
                                                            check_pos_def=True)

np.save(f"{like_product_dir}/x_ar_{cov_name}.npy", analytic_cov)
corr_analytic = so_cov.cov2corr(analytic_cov, remove_diag=True)
so_cov.plot_cov_matrix(corr_analytic, file_name=f"{plot_dir}/corr_xar")

if use_beam_covariance:

    beam_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                            cov_dir,
                                                            "analytic_beam_cov",
                                                            spectra_order=["TT", "TE", "ET", "EE"],
                                                            remove_doublon=True,
                                                            check_pos_def=False)

    analytic_cov_with_beam = analytic_cov + beam_cov

    pspy_utils.is_pos_def(analytic_cov_with_beam)
    pspy_utils.is_symmetric(analytic_cov_with_beam)

    np.save(f"{like_product_dir}/x_ar_{cov_name}.npy", analytic_cov_with_beam)

    corr_analytic_cov_with_beam = so_cov.cov2corr(analytic_cov_with_beam, remove_diag=True)
    so_cov.plot_cov_matrix(corr_analytic_cov_with_beam, file_name=f"{plot_dir}/corr_xar_with_beam")


# This part compare the analytic covariance with the montecarlo covariance
# In particular it produce plot of all diagonals of the matrix with MC vs analytics
# We use our usual javascript visualisation tools

compare_with_sims = False
if compare_with_sims:
    mc_dir = "montecarlo"

    cov_plot_dir = "plots/full_covariance"
    pspy_utils.create_directory(cov_plot_dir)
    multistep_path = d["multistep_path"]

    full_mc_cov = np.load("%s/cov_restricted_all_cross.npy" % mc_dir)

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
    n_bins = len(bin_hi)

    os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, cov_plot_dir))
    file = "%s/covariance.html" % (cov_plot_dir)
    g = open(file, mode="w")
    g.write('<html>\n')
    g.write('<head>\n')
    g.write('<title> covariance </title>\n')
    g.write('<script src="multistep2.js"></script>\n')
    g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
    g.write('<style> \n')
    g.write('body { text-align: center; } \n')
    g.write('img { width: 100%; max-width: 1200px; } \n')
    g.write('</style> \n')
    g.write('</head> \n')
    g.write('<body> \n')
    g.write('<div class=sub>\n')

    size = int(full_analytic_cov.shape[0] / n_bins)
    count = 0
    for ispec in range(-size + 1, size):

        rows, cols = np.indices(full_mc_cov.shape)
        row_vals = np.diag(rows, k = ispec * n_bins)
        col_vals = np.diag(cols, k = ispec * n_bins)
        mat = np.ones(full_mc_cov.shape)
        mat[row_vals, col_vals] = 0

        str = "cov_diagonal_%03d.png" % (count)

        plt.figure(figsize=(12,8))
        plt.subplot(1, 2, 1)
        plt.plot(np.log(np.abs(full_analytic_cov.diagonal(ispec * n_bins))))
        plt.plot(np.log(np.abs(full_mc_cov.diagonal(ispec * n_bins))), '.')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.imshow(np.log(np.abs(full_analytic_cov * mat)))
        plt.savefig(f"{cov_plot_dir}/{str}")
        plt.clf()
        plt.close()

        g.write('<div class=sub>\n')
        g.write('<img src="' + str + '"  /> \n')
        g.write('</div>\n')

        count+=1

    g.write('</body> \n')
    g.write('</html> \n')
    g.close()
