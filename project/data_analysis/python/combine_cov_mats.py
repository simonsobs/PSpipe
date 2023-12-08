"""
This script combines the different covariances together.
It does it for two types of objects: it combines the xar covariance together and it
combines the full covariance matrices together to then write on disk its decomposition into a block format.
(the difference between xar and full cov is that in the former we removed doublon, such as TE, ET for same array and survey)

The first operation is used for producing the covariance entering the likelihood while the second is useful for other scripts
such as the ones computing null test.
Technically we could avoid doing both but for now we judge this redundancy to be convenient

We consider two different ways for correcting the covariance using simulation
mc1: only replace the diagonal element of the covariance matrix
mc2: keep the analytcal structure of the correlation matrix and rescale the entire matrix by using the montecarlo estimated diagonal
"""

import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, so_cov, pspy_utils
from pspipe_utils import covariance, pspipe_list, log
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

cov_dir = "covariances"
mc_dir = "montecarlo"
plot_dir = "plots/x_ar_cov"

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

mc_correction = True
include_beam = True
include_leakage = True
only_diag_corrections = True

cov_name = "cov"
    
full_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                        cov_dir,
                                                        "analytic_cov",
                                                        spectra_order=spectra)
                                                                 
xar_cov = np.load(f"{cov_dir}/x_ar_analytic_cov.npy")

if mc_correction:

    log.info(f"include monte carlo correction (only_diag_corrections={only_diag_corrections})")


    full_mc_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                               mc_dir,
                                                               "mc_cov",
                                                               spectra_order=spectra)
    
    full_cov = covariance.correct_analytical_cov(full_cov,
                                                 full_mc_cov,
                                                 only_diag_corrections=only_diag_corrections)
                                                 
    xar_mc_cov = np.load(f"{mc_dir}/x_ar_mc_cov.npy")
    
    xar_cov = covariance.correct_analytical_cov(xar_cov,
                                                xar_mc_cov,
                                                only_diag_corrections=only_diag_corrections)


    if only_diag_corrections == True:
        cov_name += "_mc1"
    else:
        cov_name += "_mc2"


if include_beam:

    log.info(f"include beam correction")

    full_beam_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                 cov_dir,
                                                                "beam_cov",
                                                                 spectra_order=spectra)
                                                            
    full_cov =  full_cov + full_beam_cov
    
    xar_beam_cov = np.load(f"{cov_dir}/x_ar_beam_cov.npy")
    xar_cov = xar_cov + xar_beam_cov
    cov_name += "_beam"

if include_leakage:

    log.info(f"include leakage beam correction")

    full_leak_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                 cov_dir,
                                                                 "leakage_cov",
                                                                 spectra_order=spectra)

    full_cov =  full_cov  + full_leak_cov
    
    xar_leak_cov = np.load(f"{cov_dir}/x_ar_leakage_cov.npy")
    xar_cov = xar_cov + xar_leak_cov

    cov_name += "_leak"


full_corr = so_cov.cov2corr(full_cov, remove_diag=True)
so_cov.plot_cov_matrix(full_corr, file_name=f"{plot_dir}/full_{cov_name}")

x_ar_corr = so_cov.cov2corr(xar_cov, remove_diag=True)
so_cov.plot_cov_matrix(x_ar_corr, file_name=f"{plot_dir}/xar_{cov_name}")

np.save(f"{cov_dir}/x_ar_{cov_name}.npy", xar_cov)

_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
n_bins = len(lb)

full_cov_dict = covariance.full_cov_to_cov_dict(full_cov,
                                                spec_name_list,
                                                n_bins,
                                                spectra_order=spectra)
                                                
covariance.cov_dict_to_file(full_cov_dict,
                            spec_name_list,
                            cov_dir,
                            cov_type=cov_name,
                            spectra_order=spectra)
