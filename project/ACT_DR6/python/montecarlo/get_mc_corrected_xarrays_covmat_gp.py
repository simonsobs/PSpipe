description = """
This script corrects an analytic covariance matrix using a monte carlo covariance
matrix. Both matrices are inferred from the paramfile by standard names. It
uses eigenspectrum GP smoothing for now.

It is short enough that it should always run in a one-shot job, so it 
accepts no arguments other than paramfile.
"""

import numpy as np
from pspipe_utils import covariance as psc, pspipe_list, log
from pspy import so_dict, pspy_utils
import matplotlib.pyplot as plt

import argparse
import os

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("paramfile", type=str,
                    help="Filename (full or relative path) of paramfile to use")
parser.add_argument("--iStart", type=int, default=None,
                    help="Correct using the cov that used only use these simulations")
parser.add_argument("--iStop", type=int, default=None,
                    help="Correct using the cov that used only use these simulations")
parser.add_argument("--plot-all", action='store_true', dest='plot_all',
                    help="Make plots of the GP fits for both the main and block diagonals")
parser.add_argument("--plot-diag", action='store_true', dest='plot_diag',
                    help="Make plots of the GP fits for the main diagonals only")
args = parser.parse_args()

plot_all = args.plot_all
plot_diag = args.plot_diag
if plot_all and plot_diag:
    raise ValueError('plot-all and plot-diag cannot be both True')

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

covariances_dir = "covariances"
plot_dir = "plots/x_ar_cov"

if plot_all or plot_diag:
    pspy_utils.create_directory(plot_dir)
    if not plot_diag:    
        pspy_utils.create_directory(plot_dir + '/off_diags')

binning_file = d["binning_file"]
lmax = d["lmax"]

# the GP is only fit over the bins that survive the below cuts
spectra_cuts = {
    "dr6_pa4_f220": {"T": [975, lmax], "P": [lmax, lmax]},
    "dr6_pa5_f090": {"T": [975, lmax], "P": [475, lmax]},
    "dr6_pa5_f150": {"T": [775, lmax], "P": [475, lmax]},
    "dr6_pa6_f090": {"T": [975, lmax], "P": [475, lmax]},
    "dr6_pa6_f150": {"T": [575, lmax], "P": [475, lmax]},
    }
only_TT_map_set = ["dr6_pa4_f220"]

if args.iStart is not None:
    iStart = args.iStart
    iStop = args.iStop

ana_cov = np.load(os.path.join(covariances_dir, "x_ar_analytic_cov.npy"))
if args.iStart is not None:
    mc_cov = np.load(os.path.join(covariances_dir, f"x_ar_mc_cov_{iStart}_{iStop}.npy"))
else:
    mc_cov = np.load(os.path.join(covariances_dir, "x_ar_mc_cov.npy"))

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_mean)

# need the errors on the flattened analytic matrix diagonal
if args.iStart is not None:
    var_mc_cov_anaflat = np.load(os.path.join(covariances_dir, f"var_x_ar_mc_cov_anaflat_{iStart}_{iStop}.npy"))
else:
    var_mc_cov_anaflat = np.load(os.path.join(covariances_dir, "var_x_ar_mc_cov_anaflat.npy"))

# we want the indices of each block that survive data cuts.
# we will use these to smooth each block.
bin_out_dict, all_indices = psc.get_indices(bin_low,
                                            bin_high,
                                            bin_mean,
                                            spec_name_list,
                                            spectra_cuts=spectra_cuts,
                                            spectra_order=spectra,
                                            selected_spectra=spectra,
                                            only_TT_map_set=only_TT_map_set)

# for each block (NOTE: assuming remove_doublon=True), get the indices, if any.
# NOTE: important to match indexing scheme: iterate over spectra first, then
# crosses
idx_arrs_by_block = []
keys = []
bin_idx = 0
for s, spec in enumerate(spectra):
    for sid, name in enumerate(spec_name_list):
        na, nb = name.split("x")
        if (na == nb) & (spec == "ET" or spec == "BT" or spec == "BE"):
            continue

        key = (name, spec)
        if key in bin_out_dict:
            id_start = bin_idx * n_bins
            idxs_into_cut_vec = bin_out_dict[key][0]
            idxs_into_full_vec = all_indices[idxs_into_cut_vec] # all_indices are after the cuts
            idxs_into_block = idxs_into_full_vec - id_start

            # sanity check
            alt_idxs_into_block = np.intersect1d(bin_mean, bin_out_dict[key][1], return_indices=True)[1]
            assert np.all(idxs_into_block == alt_idxs_into_block), \
                f"Brute force got {idxs_into_block}, expected {alt_idxs_into_block} for {key}"
        else:
            idxs_into_block = []

        idx_arrs_by_block.append(idxs_into_block)
        keys.append(key)
        bin_idx += 1

# do the correction
corrected_mc_cov, mc_cov_anaflat, smoothed_mc_cov_anaflat, gprs = psc.correct_analytical_cov_block_diag_gp(bin_mean,
                                                                                                           ana_cov,
                                                                                                           mc_cov,
                                                                                                           var_mc_cov_anaflat=var_mc_cov_anaflat,
                                                                                                           idx_arrs_by_block=idx_arrs_by_block,
                                                                                                           return_all=True)

pspy_utils.is_pos_def(corrected_mc_cov)
pspy_utils.is_symmetric(corrected_mc_cov)

if args.iStart is not None:
    np.save(os.path.join(covariances_dir, f"x_ar_final_cov_sim_gp_{iStart}_{iStop}.npy"), corrected_mc_cov)
else:
    np.save(os.path.join(covariances_dir, "x_ar_final_cov_sim_gp.npy"), corrected_mc_cov)

# make plots
if plot_all or plot_diag:
    for i in range(len(keys)):
        name_i, spec_i = keys[i]

        if plot_diag:
            js = [i]
        else:
            js = range(i, len(keys))

        for j in js:
            name_j, spec_j = keys[j]

            if gprs[i, j] is not None:
                kern = gprs[i, j].kernel_
            else:
                kern = ''
            print(name_i, spec_i, name_j, spec_j, kern)

            sel = np.s_[i*n_bins:(i+1)*n_bins, j*n_bins:(j+1)*n_bins]
            plt.errorbar(bin_mean, np.diag(mc_cov_anaflat[sel]), np.diag(var_mc_cov_anaflat[sel])**.5, zorder=0, label='MC')
            plt.plot(bin_mean, np.diag(smoothed_mc_cov_anaflat[sel]), zorder=1, label='GP(MC)')

            # get idxs
            idxs_i = idx_arrs_by_block[i]
            idxs_j = idx_arrs_by_block[j]
            if idxs_i is not None and idxs_j is not None:
                idxs = np.intersect1d(idxs_i, idxs_j)
            elif idxs_i is None:
                idxs = idxs_j # idxs_i is "all idxs" so use idxs_j (which might also be "all idxs")
            else:
                idxs = idxs_i # idxs_j is "all idxs" so use idxs_i
            if idxs is None:
                idxs = np.arange(n_bins, dtype=int)

            if len(idxs) > 0:
                plt.axvline(bin_mean[idxs[0]], linestyle='--', color='k')
                plt.axvline(bin_mean[idxs[-1]], linestyle='--', color='k')
            if i == j:
                plt.ylim(.5, 1.5)
            else:
                plt.ylim(-.1, .1)
            plt.grid()
            plt.xlabel("l")
            plt.ylabel("ratio")
            plt.legend()
            plt.title(f"Cov({name_i}_{spec_i}, {name_j}_{spec_j})\n{kern}")
            if args.iStart is not None:
                if i == j:
                    plt.savefig(os.path.join(plot_dir, f"GP_MC_covmat_smooth_{name_i}_{spec_i}_{name_j}_{spec_j}_{iStart}_{iStop}.png"))
                else:
                    plt.savefig(os.path.join(plot_dir, 'off_diags', f"GP_MC_covmat_smooth_{name_i}_{spec_i}_{name_j}_{spec_j}_{iStart}_{iStop}.png"))
            else:
                if i == j:
                    plt.savefig(os.path.join(plot_dir, f"GP_MC_covmat_smooth_{name_i}_{spec_i}_{name_j}_{spec_j}.png"))
                else:
                    plt.savefig(os.path.join(plot_dir, 'off_diags', f"GP_MC_covmat_smooth_{name_i}_{spec_i}_{name_j}_{spec_j}.png"))
            plt.close()