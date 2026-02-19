description = """
Convert all the multiplicative parts of PSpipe into matrices.
"""
import numpy as np
import matplotlib.pyplot as plt
from pspy import so_map, so_dict, pspy_utils, so_mcm, so_mpi
from pspipe_utils import log, pspipe_list, kspace
import healpy as hp
from os.path import join as opj
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

mcm_dir = d['mcm_dir']
plot_dir = opj(d['plots_dir'], 'mcms')
spec_dir = d["spec_dir"]
pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(spec_dir)

surveys = d["surveys"]
lmax = d['lmax']
binned_mcm = d['binned_mcm']
binning_file = d['binning_file']
apply_kspace_filter = d["apply_kspace_filter"]
kspace_tf_path = d["kspace_tf_path"]
deconvolve_pixwin = d["deconvolve_pixwin"] # FIXME: this might not be one thing for all surveys etc.

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
spin_pairs = ('spin0xspin0', 'spin0xspin2', 'spin2xspin0', 'spin2xspin2_diag', 'spin2xspin2_off')
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"] # FIXME: block order assume spectra order

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

# get map-level and spectrum-level auxiliary data products related to the 
# kspace filter and pixwins
if apply_kspace_filter:
    maps, templates, filter_dicts =  {}, {}, {}
    for sv in surveys:
        maps[sv] = d[f"arrays_{sv}"] # TODO: replace with maps, arrays is confusing
        
        # FIXME: this will not work for SO LF which has a different template despite
        # being the same survey
        templates[sv] = so_map.read_map(d[f"window_kspace_{sv}_{maps[sv][0]}"])
            
        if templates[sv].pixel == "CAR":
            filter_dicts[sv] = d[f"k_filter_{sv}"]
        else:
            raise NotImplementedError('can only kspace filter CAR maps')

    if kspace_tf_path == "analytical":
        # FIXME: func assumes len(spectra) == 9
        kspace_transfer_matrix = kspace.build_analytic_kspace_filter_matrices(surveys, # FIXME: will break if any non-CAR survey
                                                                              maps,
                                                                              templates,
                                                                              filter_dicts,
                                                                              binning_file, # FIXME: assumes same binning all maps
                                                                              lmax)
    else:
        kspace_transfer_matrix = {}
        for spec_name in spec_name_list:
            # FIXME: func assumes len(spectra) == 9
            # FIXME: script assumes (below) same spectra ordering as what made these matrices
            kspace_transfer_matrix[spec_name] = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy", allow_pickle=True)

    for k, v in kspace_transfer_matrix.items():
        if np.count_nonzero(v.diagonal() == 0):
            log.info(f'WARNING: 0 in kspace_transfer_matrix {k}')

    # this will be used in the old covariance computation
    for spec_name in spec_name_list:
        one_d_tf = kspace_transfer_matrix[spec_name].diagonal()
        cov_T_E_only = d["cov_T_E_only"]
        if cov_T_E_only == True: one_d_tf = one_d_tf[:4 * n_bins]
        np.savetxt(f"{spec_dir}/one_dimension_kspace_tf_{spec_name}.dat", one_d_tf)

pixwins = {}
for sv in surveys:
    if d[f"pixwin_{sv}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
        # this is a crude approximation. really, it would be something like
        # Bbl @ (pw_l)^2 C_l, so it can't be easily decoupled
        pw_l = hp.pixwin(d[f"pixwin_{sv}"]["nside"])
        _, pw_b = pspy_utils.naive_binning(np.arange(len(pw_l)), pw_l, binning_file, lmax)
        pixwins[sv] = pw_b

# now get the operators
nspec, sv1_list, m1_list, sv2_list, m2_list = pspipe_list.get_spectra_list(d)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=nspec - 1)
log.info(f"[Rank {so_mpi.rank}] Number of operators to compute: {len(subtasks)}")
for task in subtasks:
    sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]
    spec_name = f"{sv1}_{m1}x{sv2}_{m2}"

    log.info(f'[Rank {so_mpi.rank}] Calculating operators for {spec_name}')
    
    # get the mbl_inv for this array cross
    mbl_inv = np.load(opj(f"{mcm_dir}", f"{spec_name}_mode_coupling_inv.npy"))

    # need to splice mbl_inv into spectra-ordered arrays. pseudo2datavec is a 
    # sparse, two-level dictionary of arrays, so the order of spectra also
    # doesn't matter
    #
    # copy blocks for safety since we might modify individual blocks below
    pseudo2datavec = so_mcm.get_spec2spec_sparse_dict_mat_from_spin2spin_array(mbl_inv, spectra, copy=True)

    # get the inv_kspace matrix for this array cross, if necessary
    if apply_kspace_filter:
        inv_kspace_mat = np.linalg.inv(kspace_transfer_matrix[spec_name]) 

        # apply the inv_kspace matrix to mbl_inv to get data operator. don't
        # need to copy because just being used in math
        # FIXME: script assumes same spectra ordering as what made these matrices
        inv_kspace_mat = so_mcm.get_spec2spec_sparse_dict_mat_from_dense_mat(inv_kspace_mat, spectra)
        pseudo2datavec = so_mcm.sparse_dict_mat_matmul_sparse_dict_mat(inv_kspace_mat, pseudo2datavec)

    # get the pixwin for healpix, if necessary
    # FIXME: put pixwin in mcm / forward model
    if d[f"pixwin_{sv1}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
        for row, col_dict in pseudo2datavec.items():
            for col in col_dict:
                pseudo2datavec[row][col] /= pixwins[sv1][:, None] # apply on the left
    if d[f"pixwin_{sv2}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
        for row, col_dict in pseudo2datavec.items():
            for col in col_dict:
                pseudo2datavec[row][col] /= pixwins[sv2][:, None] # apply on the left

    # save and plot
    np.save(opj(f'{mcm_dir}', f'pseudo2datavec_{spec_name}.npy'), pseudo2datavec)
    
    plt.figure(figsize=(10, 8))
    pseudo2datavec = so_mcm.sparse_dict_mat2dense_array(pseudo2datavec, np.float32)
    plt.imshow(np.log(np.abs(pseudo2datavec)), aspect=100)
    plt.xticks([pseudo2datavec.shape[1] * (2 * i + 1) / 18 for i in range(9)], spectra)
    plt.yticks([pseudo2datavec.shape[0] * (2 * i + 1) / 18 for i in range(9)], spectra)
    plt.colorbar()
    plt.title(f'pseudo2datavec {spec_name}')
    plt.savefig(opj(f'{plot_dir}', f'pseudo2datavec_{spec_name}'), bbox_inches='tight')
    plt.close()
