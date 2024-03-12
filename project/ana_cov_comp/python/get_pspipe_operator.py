"""
So far we have produced the covariance matrix of the measured pseudospectra, 
but we want the covariance matrix of the power spectra. The power spectra are
(mostly) equivalent to a linear operator acting on the measured pseudospectra.
Thus, if this operator is F, and the pseudocovariance block is P, then the 
power spectrum covariance block is F @ P @ F.T. This script produces F
from other PSpipe products. It includes: mode-decoupling, binning (with possible)
Dl factors, and kspace deconvolving.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pspy import so_map, so_dict, pspy_utils, so_mcm
from pspipe_utils import log, pspipe_list, kspace, covariance as psc

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

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

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

# get the binning matrix
Pbl = psc.get_binning_matrix(bin_lo, bin_hi, lmax)

for spec1 in spec_list:
    log.info(f'Calculating matrix for {spec1}')
    
    # get the Mbb_inv matrix for this array cross
    M_inv, _ = so_mcm.read_coupling(prefix=f'{mcms_dir}/{spec1}', spin_pairs=spin_pairs)

    # apply the binning matrix to it to get Pbl_Minv
    Pbl_Minv = psc.get_Pbl_Minv_matrix(Pbl, M_inv) 

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