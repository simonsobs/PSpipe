description = """
This script uses the covariance matrix blocks to form a cross array covariance matrix
Note that for the ET - BT - BE blocks, we do not include any same array, same survey spectra, since for
these guys XY = YX and therefore are already included in the TE - TB - EB.
The x_ar cov is organised as TT - TE -TB - ET - BT - EE - EB - BE -BB for cov_T_E_only = False, TT - TE - ET - EE otherwise
then each of these blocks contains all x_array terms e.g pa5_f090xpa5_f090, pa5_f090xpa5_f150 ..
"""

from pspy import so_dict, pspy_utils, so_mcm, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, log
import numpy as np

import sys
from os.path import join as opj
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--check-pos-def', action='store_true', # default False, type bool
                    help='Check if the matrix is positive definite and symmetric.')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

lmax = d['lmax']
type = d['type']

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes_for_cov = ["TT", "TE", "ET", "EE"]
else:
    modes_for_cov = spectra

bestfit_dir = d["best_fits_dir"]
mcm_dir = d['mcm_dir']
binned_mcm = d['binned_mcm']
spec_dir = d['spec_dir']
cov_dir = d['cov_dir']
plot_dir = opj(d['plots_dir'], 'covariances')
pspy_utils.create_directory(plot_dir)

# TODO: clean up with covariance.py functions
data_dict = {}
theory_dict = {}
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_", from_spec_nullgroups=d['spectra_list_from_spec_nullgroups'])
for spec_name in spec_name_list:
    _, _data_dict = so_spectra.read_ps(opj(spec_dir, f"{type}_{spec_name}_cross.dat"),
                                       spectra=spectra, return_type=type,
                                       return_dtype=np.float64)

    for spec in spectra:
        data_dict[spec_name, spec] = _data_dict[spec]
 
    _l, _theory_dict = so_spectra.read_ps(opj(bestfit_dir, f"cmb_and_fg_{spec_name}.dat"),
                                         spectra=spectra, return_type=type,
                                         return_dtype=np.float64)
    assert _l[0] == 2, f'expected l[0]=2, got {_l[0]=}'

    for spec, theoryl in _theory_dict.items():
        _theory_dict[spec] = theoryl[:lmax] # trim to match with Bbl
    
    Bbl = np.load(opj(mcm_dir, f'{spec_name}_Bbl.npy'))
    if binned_mcm:
        spec2spec2Bbl = so_mcm.get_spec2spec_sparse_dict_mat_from_spin2spin_array(Bbl, spectra)
    else:
        spec2spec2Bbl = so_mcm.get_block_diagonal_sparse_dict_mat_from_array(Bbl, spectra)

    _theory_dict = so_mcm.sparse_dict_mat_matmul_sparse_dict_vec(spec2spec2Bbl, _theory_dict)

    for spec in spectra:
        theory_dict[spec_name, spec] = _theory_dict[spec]

x_ar_data_vec = covariance.spec_dict_to_full_vec(data_dict, 
                                                 spec_name_list,
                                                 spectra_order=spectra,
                                                 remove_doublon=True)

x_ar_theory_vec = covariance.spec_dict_to_full_vec(theory_dict, 
                                                   spec_name_list,
                                                   spectra_order=spectra,
                                                   remove_doublon=True)

np.save(opj(spec_dir, "x_ar_data_vec.npy"), x_ar_data_vec)
np.save(opj(bestfit_dir, "x_ar_theory_vec.npy"), x_ar_theory_vec)

# FIXME: clean up posdef stuff
def check_corr(corr, check_pos_def=True):
    # do some checks even if not pos def (which is more expensive)
    if not np.allclose(corr, corr.T, rtol=1e-5, atol=1e-5): # single precision
        raise ValueError('The corr is not symmetric, so the matrix still is invalid')

    if check_pos_def:
        np.linalg.cholesky(corr + np.eye(corr.shape[0])) # this will fail if not pos def
    else:
        if not np.all(np.abs(corr) < 1):
            raise ValueError('Did not check pos def but still some corr are >=1, so the matrix still is invalid')

# do the operations
log.info(f"create x array cov mat from analytic cov block")

x_ar_analytic_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                 cov_dir,
                                                                 "analytic_cov",
                                                                 spectra_order=modes_for_cov,
                                                                 remove_doublon=True,
                                                                 check_pos_def=False)

x_ar_analytic_corr = so_cov.cov2corr(x_ar_analytic_cov, remove_diag=True)
check_corr(x_ar_analytic_corr, check_pos_def=args.check_pos_def)

# symmetrize for numeric crap
x_ar_analytic_cov = (x_ar_analytic_cov + x_ar_analytic_cov.T)/2
x_ar_analytic_corr = (x_ar_analytic_corr + x_ar_analytic_corr.T)/2

np.save(opj(cov_dir, "x_ar_analytic_cov.npy"), x_ar_analytic_cov)
so_cov.plot_cov_matrix(x_ar_analytic_corr, file_name=opj(plot_dir, "xar_analytic_corr"))

if d.get("use_beam_covariance"):
    log.info(f"create x array beam cov mat from beam cov block")

    x_ar_beam_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                 cov_dir,
                                                                 "beam_cov",
                                                                 spectra_order=modes_for_cov,
                                                                 remove_doublon=True,
                                                                 check_pos_def=False)

    x_ar_beam_corr = so_cov.cov2corr(x_ar_beam_cov, remove_diag=True)
    check_corr(x_ar_beam_corr, check_pos_def=args.check_pos_def)

    # symmetrize for numeric crap
    x_ar_beam_cov = (x_ar_beam_cov + x_ar_beam_cov.T)/2
    x_ar_beam_corr = (x_ar_beam_corr + x_ar_beam_corr.T)/2
    
    np.save(opj(cov_dir, "x_ar_beam_cov.npy"), x_ar_beam_cov)
    so_cov.plot_cov_matrix(x_ar_beam_corr, file_name=opj(plot_dir, "xar_beam_corr"))