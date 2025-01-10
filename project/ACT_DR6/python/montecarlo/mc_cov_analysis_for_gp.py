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
parser.add_argument('--iStart', type=int, default=None,
                    help='Only use these simulations to build the cov')
parser.add_argument('--iStop', type=int, default=None,
                    help='Only use these simulations to build the cov')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

type = d["type"]
iStart = d["iStart"]
iStop = d["iStop"]
if args.iStart is not None:
    iStart = args.iStart
    iStop = args.iStop
sim_idxs = range(iStart, iStop + 1) # iStart + (iStop + 1 - iStart)
N = len(sim_idxs)

sim_spec_dir = d["sim_spec_dir"]
covariances_dir = "covariances"

pspy_utils.create_directory(covariances_dir)

ana_cov = np.load(os.path.join(covariances_dir, 'x_ar_analytic_cov.npy'))
inv_sqrt_ana_cov  = utils.eigpow(ana_cov, -0.5)

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
full_vec_anaflat_list = []

for iii in sim_idxs:
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
    full_vec_anaflat_list.append(inv_sqrt_ana_cov @ full_vec_iii)

mean_full_vec = np.mean(full_vec_list, axis=0)
mean_full_vec_anaflat = np.mean(full_vec_anaflat_list, axis=0)

# do another pass to get the mean mc_covs. we don't store them all for memory reasons.
# mean mc_cov includes Bessel's correction
mean_mc_cov = np.zeros((mean_full_vec.size, mean_full_vec.size))
mean_mc_cov_anaflat = np.zeros((mean_full_vec_anaflat.size, mean_full_vec_anaflat.size))

@numba.njit(parallel=True)
def add_term_to_mc_cov(mc_cov, samp_vec, mean_vec, N):
    delta_vec = samp_vec - mean_vec
    term = (N/(N-1)) * np.outer(delta_vec, delta_vec)
    mc_cov += term

for iii in sim_idxs:
    if iii % 100 == 0:
        log.info(iii)

    add_term_to_mc_cov(mean_mc_cov, full_vec_list[iii], mean_full_vec, N)
    add_term_to_mc_cov(mean_mc_cov_anaflat, full_vec_anaflat_list[iii],
                       mean_full_vec_anaflat, N)

mean_mc_cov /= N
pspy_utils.is_symmetric(mean_mc_cov)

mean_mc_cov_anaflat /= N
pspy_utils.is_symmetric(mean_mc_cov_anaflat)

# do a third pass to get errors in these quantities/
# variance includes Bessel's correction, and then the variance of the mean over sims
var_mc_cov = np.zeros((mean_full_vec.size, mean_full_vec.size))
var_mc_cov_anaflat = np.zeros((mean_full_vec_anaflat.size, mean_full_vec_anaflat.size)) 

@numba.njit(parallel=True)
def add_term_to_var_mc_cov(var_mc_cov, samp_vec, mean_vec, mean_mc_cov, N):
    delta_vec = samp_vec - mean_vec
    term = (N/(N-1)) * np.outer(delta_vec, delta_vec)
    var_mc_cov += (term - mean_mc_cov)**2

for iii in sim_idxs:
    if iii % 100 == 0:
        log.info(iii)

    add_term_to_var_mc_cov(var_mc_cov, full_vec_list[iii], mean_full_vec,
                           mean_mc_cov, N)
    add_term_to_var_mc_cov(var_mc_cov_anaflat, full_vec_anaflat_list[iii],
                           mean_full_vec_anaflat, mean_mc_cov_anaflat, N)

var_mc_cov /= N * (N-1)
pspy_utils.is_symmetric(var_mc_cov)

var_mc_cov_anaflat /= N * (N-1)
pspy_utils.is_symmetric(var_mc_cov_anaflat)

# save matrices
if args.iStart is not None:
    fn = f"x_ar_mc_cov_{iStart}_{iStop}.npy"
    var_fn = f"var_x_ar_mc_cov_{iStart}_{iStop}.npy"
else:
    fn = "x_ar_mc_cov.npy"
    var_fn = "var_x_ar_mc_cov.npy"
np.save(os.path.join(covariances_dir, fn), mean_mc_cov)
np.save(os.path.join(covariances_dir, var_fn), var_mc_cov)

if args.iStart is not None:
    fn = f"x_ar_mc_cov_anaflat_{iStart}_{iStop}.npy"
    var_fn = f"var_x_ar_mc_cov_anaflat_{iStart}_{iStop}.npy"
else:
    fn = "x_ar_mc_cov_anaflat.npy"
    var_fn = "var_x_ar_mc_cov_anaflat.npy"
np.save(os.path.join(covariances_dir, fn), mean_mc_cov_anaflat)
np.save(os.path.join(covariances_dir, var_fn), var_mc_cov_anaflat)
