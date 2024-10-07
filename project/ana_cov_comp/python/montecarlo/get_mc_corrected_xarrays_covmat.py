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
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--iStart', type=int, default=None,
                    help='Only use these simulations')
parser.add_argument('--iStop', type=int, default=None,
                    help='Only use these simulations')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

covariances_dir = d['covariances_dir']
plot_dir = d["plot_dir"] + "/covariances"
pspy_utils.create_directory(plot_dir)

binning_file = d["binning_file"]
lmax = d["lmax"]
spectra_cuts = d['spectra_cuts']
only_TT_map_set = d['only_TT_map_set']

if args.iStart is not None:
    iStart = args.iStart
    iStop = args.iStop

ana_cov = np.load(os.path.join(covariances_dir, 'x_ar_analytic_cov.npy'))
if args.iStart is not None:
    mc_cov = np.load(os.path.join(covariances_dir, f'x_ar_mc_cov_{iStart}_{iStop}.npy'))
else:
    mc_cov = np.load(os.path.join(covariances_dir, 'x_ar_mc_cov.npy'))

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_mean)

# need the errors on the flattened analytic matrix diagonal
if args.iStart is not None:
    var_ana_cov_flat = np.load(os.path.join(covariances_dir, f'var_x_ar_mc_cov_anaflat_{iStart}_{iStop}.npy'))
else:
    var_ana_cov_flat = np.load(os.path.join(covariances_dir, 'var_x_ar_mc_cov_anaflat.npy'))
var_eigenspectrum_ratios_by_block = np.split(np.diag(var_ana_cov_flat), var_ana_cov_flat.shape[0] // n_bins)

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
                f'Brute force got {idxs_into_block}, expected {alt_idxs_into_block} for {key}'
        else:
            idxs_into_block = []

        idx_arrs_by_block.append(idxs_into_block)
        keys.append(key)
        bin_idx += 1

# do the correction
corr_cov, res_diag, smoothed_res_diag, gprs = psc.correct_analytical_cov_eigenspectrum_ratio_gp(bin_mean,
                                                                                                ana_cov,
                                                                                                mc_cov,
                                                                                                var_eigenspectrum_ratios_by_block=var_eigenspectrum_ratios_by_block,
                                                                                                idx_arrs_by_block=idx_arrs_by_block,
                                                                                                return_all=True)

pspy_utils.is_pos_def(corr_cov)
pspy_utils.is_symmetric(corr_cov)

if args.iStart is not None:
    np.save(os.path.join(covariances_dir, f'x_ar_final_cov_sim_{iStart}_{iStop}.npy'), corr_cov)
else:
    np.save(os.path.join(covariances_dir, 'x_ar_final_cov_sim.npy'), corr_cov)

# make plots
for i, (name, spec) in enumerate(keys):
    if gprs[i] is not None:
        kern = gprs[i].kernel_
    else:
        kern = ''
    print(name, spec, kern)

    sel = np.s_[i*n_bins:(i+1)*n_bins]
    plt.errorbar(bin_mean, res_diag[sel], np.diag(var_ana_cov_flat)[sel]**.5, zorder=0, label='MC')
    plt.plot(bin_mean, smoothed_res_diag[sel], zorder=1, label='GP(MC)')
    if len(idx_arrs_by_block[i]) > 0:
        plt.axvline(bin_mean[idx_arrs_by_block[i][0]], linestyle='--', color='k')
        plt.axvline(bin_mean[idx_arrs_by_block[i][-1]], linestyle='--', color='k')
    plt.ylim(.5, 1.5)
    plt.grid()
    plt.xlabel('l')
    plt.ylabel('ratio')
    plt.legend()
    plt.title(f'Cov({name}_{spec}, {name}_{spec})\n{kern}')
    if args.iStart is not None:
        plt.savefig(os.path.join(plot_dir, f'GP_MC_covmat_smooth_{name}_{spec}_{iStart}_{iStop}.png'))
    else:
        plt.savefig(os.path.join(plot_dir, f'GP_MC_covmat_smooth_{name}_{spec}.png'))
    plt.close()