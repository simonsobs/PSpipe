description = """
So far we have produced the covariance matrix of the measured pseudospectra, 
but we want the covariance matrix of the power spectra. The power spectra are
(mostly) equivalent to a linear operator acting on the measured pseudospectra.
Thus, if this operator is F, and the pseudocovariance block is P, then the 
power spectrum covariance block is F @ P @ F.T. This script produces F
from other PSpipe products. It includes: mode-decoupling, binning (with possible)
Dl factors, and kspace deconvolving.

It is short enough that it should always run in a one-shot job, so it 
accepts no arguments other than paramfile.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from pspy import so_map, so_dict, pspy_utils, so_mcm
from pspipe_utils import log, pspipe_list, kspace, covariance as psc
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

mcms_dir = d['mcms_dir']
pspipe_operators_dir = d['pspipe_operators_dir']
plot_dir = os.path.join(d['plot_dir'], 'pspipe_operators')
pspy_utils.create_directory(pspipe_operators_dir)
pspy_utils.create_directory(plot_dir)

lmax = d['lmax']
binned_mcm = d['binned_mcm']
binning_file = d['binning_file']
kspace_tf_path = d['kspace_tf_path']

assert not binned_mcm, 'script only works if binned_mcm is False!' # FIXME

spec_list = pspipe_list.get_spec_name_list(d, delimiter='_')  # unrolled fields
spin_pairs = ('spin0xspin0', 'spin0xspin2', 'spin2xspin0', 'spin2xspin2')
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

# get the binning matrix
Pbl = psc.get_binning_matrix(bin_lo, bin_hi, lmax)
Pbl_pol = block_diag(Pbl, Pbl, Pbl, Pbl)

for spec1 in spec_list:
    log.info(f'Calculating matrix for {spec1}')
    
    # get the Mbb_inv matrix for this array cross
    M_inv, _ = so_mcm.read_coupling(prefix=f'{mcms_dir}/{spec1}', spin_pairs=spin_pairs)

    # compute P_{bl} @ Minv_{ll'}, the binning operator applied to the inverse
    # MCM. Better to do this block-wise than materialize the full unbinned MCM
    # across all polarizations.
    M00 = Pbl @ M_inv['spin0xspin0']
    M02 = Pbl @ M_inv['spin0xspin2']
    M20 = Pbl @ M_inv['spin2xspin0']
    M22 = Pbl_pol @ M_inv['spin2xspin2']
    
    assert spectra == ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"] # FIXME: block order assume spectra order
    Pbl_Minv = block_diag(M00, M02, M02, M20, M20, M22)

    # get the inv_kspace matrix for this array cross
    if kspace_tf_path == 'analytical':
        surveys = d['surveys']

        arrays, templates, filter_dicts =  {}, {}, {}
        for sv in surveys:
            arrays[sv] = d[f'arrays_{sv}']
            templates[sv] = so_map.read_map(d[f'window_T_{sv}_{arrays[sv][0]}']) # FIXME: assumes all templates are the same within a survey
            filter_dicts[sv] = d[f'k_filter_{sv}']

        kspace_transfer_matrix = kspace.build_analytic_kspace_filter_matrices(
            surveys, arrays, templates, filter_dicts, binning_file, lmax
            )[spec1]
    else:
        kspace_transfer_matrix = np.load(f'{kspace_tf_path}/kspace_matrix_{spec1}.npy')

    inv_kspace_mat = np.linalg.inv(kspace_transfer_matrix) 

    # apply the inv_kspace matrix to Pbl_Minv to get Finv_Pbl_Minv
    Finv_Pbl_Minv = inv_kspace_mat @ Pbl_Minv 

    np.save(f'{pspipe_operators_dir}/Finv_Pbl_Minv_{spec1}.npy', Finv_Pbl_Minv) # fin

    plt.figure(figsize=(10, 8))
    plt.imshow(np.log(np.abs(Finv_Pbl_Minv[:, ::4])), aspect=25)
    plt.colorbar()
    plt.title(f'Finv_Pbl_Minv {spec1}')
    plt.savefig(f'Finv_Pbl_Minv_{spec1}', bbox_inches='tight')
    plt.close()