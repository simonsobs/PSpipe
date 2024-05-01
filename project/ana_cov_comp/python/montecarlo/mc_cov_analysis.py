description = """
This script converts monte-carlo power spectra into monte-carlo 
covariance matrices. It infers filenames directly from the paramfile.
In preparation for the sim-based correction, it also computes the 
errorbars on the monte-carlo covariance matrix in the eigenbasis
of the analytic matrix, if the analytic matrix exists.

It is short enough that it should always run in a one-shot job, so it 
accepts no arguments other than paramfile.
"""

from pspy import pspy_utils, so_dict, so_spectra
from pspipe_utils import pspipe_list, covariance, log
import numpy as np
import numba
from pixell import utils

import os
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

type = d["type"]
iStart = d["iStart"]
iStop = d["iStop"]

sim_spec_dir =  d['data_dir'] + "sim_spectra"
covariances_dir = d['covariances_dir']

pspy_utils.create_directory(covariances_dir)

try:
    ana_cov = np.load(os.path.join(covariances_dir, 'x_ar_analytic_cov.npy'))
    inv_sqrt_ana_cov  = utils.eigpow(ana_cov, -0.5)
except FileNotFoundError:
    ana_cov = None
    log.info('No analytic cov, proceeding without it. Note, this means cannot '
             'compute explicit errors on mc covmat for sim-based correction. '
             'Instead, they will be inferred from the MC scatter.')

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes_for_cov = ["TT", "TE", "ET", "EE"]
else:
    modes_for_cov = spectra


spec_list = pspipe_list.get_spec_name_list(d, delimiter="_")

log.info(f"we start by constructing block mc covariances")

# We will iterate over sims first. for each sim, we will construct the full
# data vector and covariance matrix, and optionally the same for a transformed
# data vector depending on if the paramfile supplies an analytic covariance 
# matrix A (if d is original data vector, we also transform it into 
# (O @ E_A**-0.5 @ O.T) @ d, where O and E_A are the eigenvectors and eigen-
# values of A, respectively). In both cases, we also measure the error in 
# the covariance matrix estimate (the variance of its elements over the sims).
# 
# We compute the second transformation because it is used to correct the 
# analytic covariance by the montecarlo. The correction may need the error
# on the transformed quantity, and it is cleanest to just measure it directly.

# do one pass to store all the simulated data vectors
full_vec_list = []
if ana_cov is not None:
    full_vec_anaflat_list = []

for iii in range(iStart, iStop + 1):
    if iii % 100 == 0:
        log.info(iii)
    spec_dict = {}
    for sid1, name1 in enumerate(spec_list):
        # load the sim and populate the spec_dict
        na, nb = name1.split("x")
        spec_name_cross_iii = f"{type}_{na}x{nb}_cross_%05d" % iii
        lb, ps_iii = so_spectra.read_ps(os.path.join(sim_spec_dir, f'{spec_name_cross_iii}.dat'), spectra=spectra)
        for spec in modes_for_cov:
            spec_dict[name1, spec] = ps_iii[spec]
    
    # get full data vector for this sim, append to list
    full_vec_iii = covariance.spec_dict_to_full_vec(spec_dict, 
                                                    spec_list,
                                                    spectra_order=modes_for_cov,
                                                    remove_doublon=True)
    full_vec_list.append(full_vec_iii)
    if ana_cov is not None:
        full_vec_anaflat_list.append(inv_sqrt_ana_cov @ full_vec_iii)

mean_full_vec = np.mean(full_vec_list, axis=0)
if ana_cov is not None:
    mean_full_vec_anaflat = np.mean(full_vec_anaflat_list, axis=0)

# in principle, we could get the mc_cov in the same pass as the data vectors,
# but since we will need a second pass for errors in the mc_cov anyway, it is
# cleaner to get the mc_cov in this second pass
mean_mc_cov = np.zeros((mean_full_vec.size, mean_full_vec.size))
var_mc_cov = np.zeros((mean_full_vec.size, mean_full_vec.size))
if ana_cov is not None:
    mean_mc_cov_anaflat = np.zeros((mean_full_vec_anaflat.size, mean_full_vec_anaflat.size))
    var_mc_cov_anaflat = np.zeros((mean_full_vec_anaflat.size, mean_full_vec_anaflat.size)) 

@numba.njit(parallel=True)
def add_term_to_mc_cov(mc_cov, var_mc_cov, delta_vec):
    term = np.outer(delta_vec, delta_vec)
    mc_cov += term
    var_mc_cov += term**2

for iii in range(iStart, iStop + 1):
    if iii % 100 == 0:
        log.info(iii)

    delta_vec_iii = full_vec_list[iii] - mean_full_vec
    add_term_to_mc_cov(mean_mc_cov, var_mc_cov, delta_vec_iii)
    if ana_cov is not None:
        delta_vec_anaflat_iii = full_vec_anaflat_list[iii] - mean_full_vec_anaflat
        add_term_to_mc_cov(mean_mc_cov_anaflat, var_mc_cov_anaflat, delta_vec_anaflat_iii)

# mean_mc_cov: includes Bessel's correction
# var_mc_cov: includes Bessel's correction, and then the variance of the mean over sims
mean_mc_cov = mean_mc_cov / (iStop - iStart) 
var_mc_cov = 1 / (iStop - iStart) * (var_mc_cov / (iStop + 1 - iStart) - mean_mc_cov**2)

pspy_utils.is_symmetric(mean_mc_cov)
pspy_utils.is_symmetric(var_mc_cov)

np.save(os.path.join(covariances_dir, 'x_ar_mc_cov.npy'), mean_mc_cov)
np.save(os.path.join(covariances_dir, 'var_x_ar_mc_cov.npy'), var_mc_cov)

# also save each block
mean_mc_cov_dict = covariance.full_cov_to_cov_dict(mean_mc_cov,
                                                   spec_list,
                                                   spectra_order=modes_for_cov,
                                                   remove_doublon=True)
var_mc_cov_dict = covariance.full_cov_to_cov_dict(var_mc_cov,
                                                  spec_list,
                                                  spectra_order=modes_for_cov,
                                                  remove_doublon=True)

n_bins = len(ps_iii[spec]) # NOTE: assumes same for all individual measured ps
for sid1, name1 in enumerate(spec_list):
    for sid2, name2 in enumerate(spec_list):
        if sid1 > sid2: continue
        mean_mc_cov_block = np.zeros((len(modes_for_cov) * n_bins, len(modes_for_cov) * n_bins))
        var_mc_cov_block = np.zeros((len(modes_for_cov) * n_bins, len(modes_for_cov) * n_bins))
        for s1, spec1 in enumerate(modes_for_cov):
            for s2, spec2 in enumerate(modes_for_cov):
                mean_mc_cov_block[s1 * n_bins:(s1+1) * n_bins, s2 * n_bins:(s2+1) * n_bins] = mean_mc_cov_dict[name1, name2, spec1, spec2]
                var_mc_cov_block[s1 * n_bins:(s1+1) * n_bins, s2 * n_bins:(s2+1) * n_bins] = var_mc_cov_dict[name1, name2, spec1, spec2]
        np.save(os.path.join(covariances_dir, f'mc_cov_{name1}_{name2}.npy'), mean_mc_cov_block)
        np.save(os.path.join(covariances_dir, f'var_mc_cov_{name1}_{name2}.npy'), var_mc_cov_block)

if ana_cov is not None:
    # mean_mc_cov: includes Bessel's correction
    # var_mc_cov: includes Bessel's correction, and then the variance of the mean over sims
    mean_mc_cov_anaflat = mean_mc_cov_anaflat / (iStop - iStart) 
    var_mc_cov_anaflat = 1 / (iStop - iStart) * (var_mc_cov_anaflat / (iStop + 1 - iStart) - mean_mc_cov_anaflat**2)

    pspy_utils.is_symmetric(mean_mc_cov_anaflat)
    pspy_utils.is_symmetric(var_mc_cov_anaflat)

    np.save(os.path.join(covariances_dir, 'x_ar_mc_cov_anaflat.npy'), mean_mc_cov_anaflat)
    np.save(os.path.join(covariances_dir, 'var_x_ar_mc_cov_anaflat.npy'), var_mc_cov_anaflat)

    # also save each block
    mean_mc_cov_anaflat_dict = covariance.full_cov_to_cov_dict(mean_mc_cov_anaflat,
                                                               spec_list,
                                                               spectra_order=modes_for_cov,
                                                               remove_doublon=True)
    var_mc_cov_anaflat_dict = covariance.full_cov_to_cov_dict(var_mc_cov_anaflat,
                                                              spec_list,
                                                              spectra_order=modes_for_cov,
                                                              remove_doublon=True)

    n_bins = len(ps_iii[spec]) # NOTE: assumes same for all individual measured ps
    for sid1, name1 in enumerate(spec_list):
        for sid2, name2 in enumerate(spec_list):
            if sid1 > sid2: continue
            mean_mc_cov_anaflat_block = np.zeros((len(modes_for_cov) * n_bins, len(modes_for_cov) * n_bins))
            var_mc_cov_anaflat_block = np.zeros((len(modes_for_cov) * n_bins, len(modes_for_cov) * n_bins))
            for s1, spec1 in enumerate(modes_for_cov):
                for s2, spec2 in enumerate(modes_for_cov):
                    mean_mc_cov_anaflat_block[s1 * n_bins:(s1+1) * n_bins, s2 * n_bins:(s2+1) * n_bins] = mean_mc_cov_anaflat_dict[name1, name2, spec1, spec2]
                    var_mc_cov_anaflat_block[s1 * n_bins:(s1+1) * n_bins, s2 * n_bins:(s2+1) * n_bins] = var_mc_cov_anaflat_dict[name1, name2, spec1, spec2]
            np.save(os.path.join(covariances_dir, f'mc_cov_anaflat_{name1}_{name2}.npy'), mean_mc_cov_anaflat_block)
            np.save(os.path.join(covariances_dir, f'var_mc_cov_anaflat_{name1}_{name2}.npy'), var_mc_cov_anaflat_block)