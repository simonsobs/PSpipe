"""
This script uses the covariance matrix blocks to form a cross array covariance matrix
Note that for the ET - BT - BE blocks, we do not include any same array, same survey spectra, since for
these guys XY = YX and therefore are already included in the TE - TB - EB.
The x_ar cov is organised as TT - TE -TB - ET - BT - EE - EB - BE -BB for cov_T_E_only = False, TT - TE - ET - EE otherwise
then each of these blocks contains all x_array terms e.g pa5_f090xpa5_f090, pa5_f090xpa5_f150 ..
"""

from pspy import so_dict, pspy_utils, so_cov
from pspipe_utils import covariance, pspipe_list, log
import numpy as np
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes_for_cov = ["TT", "TE", "ET", "EE"]
else:
    modes_for_cov = spectra

cov_dir = "covariances"
plot_dir = "plots/x_ar_cov"

pspy_utils.create_directory(plot_dir)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

log.info(f"create x array cov mat from analytic cov block")

x_ar_analytic_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                 cov_dir,
                                                                 "analytic_cov",
                                                                 spectra_order=modes_for_cov,
                                                                 remove_doublon=True,
                                                                 check_pos_def=True)

np.save(f"{cov_dir}/x_ar_analytic_cov.npy", x_ar_analytic_cov)
x_ar_analytic_corr = so_cov.cov2corr(x_ar_analytic_cov, remove_diag=True)
so_cov.plot_cov_matrix(x_ar_analytic_corr, file_name=f"{plot_dir}/xar_analytic_corr")


if d["use_beam_covariance"]:
    log.info(f"create x array beam cov mat from beam cov block")

    x_ar_beam_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                 cov_dir,
                                                                 "beam_cov",
                                                                 spectra_order=modes_for_cov,
                                                                 remove_doublon=True,
                                                                 check_pos_def=False)

    np.save(f"{cov_dir}/x_ar_beam_cov.npy", x_ar_beam_cov)
    x_ar_beam_corr = so_cov.cov2corr(x_ar_beam_cov, remove_diag=True)
    so_cov.plot_cov_matrix(x_ar_beam_corr, file_name=f"{plot_dir}/xar_beam_corr")



