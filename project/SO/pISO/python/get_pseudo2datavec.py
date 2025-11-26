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
plot_base_dir = d["plots_base_dir"]
plot_dir = f"{plot_base_dir}/mcms/"
pspy_utils.create_directory(plot_dir)

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
            
        if d[f"pixwin_{sv}"]["pix"] == "CAR":
            filter_dicts[sv] = d[f"k_filter_{sv}"]
        else:
            raise NotImplementedError('can only kspace filter CAR maps')

    if kspace_tf_path == "analytical":
        kspace_transfer_matrix = kspace.build_analytic_kspace_filter_matrices(surveys, # FIXME: will break if any non-CAR survey
                                                                              maps,
                                                                              templates,
                                                                              filter_dicts,
                                                                              binning_file, # FIXME: assumes same binning all maps
                                                                              lmax)
    else:
        kspace_transfer_matrix = {}
        for spec_name in spec_name_list:
            kspace_transfer_matrix[spec_name] = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy", allow_pickle=True)

    for k, v in kspace_transfer_matrix.items():
        if np.count_nonzero(v.diagonal() == 0):
            log.info(f'WARNING: 0 in kspace_transfer_matrix {k}')

if d[f"pixwin_{sv}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
    pixwins = {}
    for sv in surveys:
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
    prefix = opj(f"{mcm_dir}", spec_name)
    mbl_inv = np.load(prefix + "_mode_coupling_inv.npy")

    # need to splice mbl_inv into spectra-ordered arrays
    # TODO: consider disk-space, memory (could be sparse)
    pseudo2datavec = so_mcm.get_spec2spec_array_from_spin2spin_array(mbl_inv, dense=True)

    # get the inv_kspace matrix for this array cross, if necessary
    if apply_kspace_filter:
        inv_kspace_mat = np.linalg.inv(kspace_transfer_matrix[spec_name]) 

        # apply the inv_kspace matrix to mbl_inv to get data operator
        pseudo2datavec = inv_kspace_mat @ pseudo2datavec

    # get the pixwin for healpix, if necessary
    if d[f"pixwin_{sv1}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
        pseudo2datavec /= np.tile(pixwins[sv1], 9)[:, None] # apply on the left
    if d[f"pixwin_{sv2}"]["pix"] == "HEALPIX" and deconvolve_pixwin:
        pseudo2datavec /= np.tile(pixwins[sv2], 9)[:, None] # apply on the left

    # save and plot
    np.save(opj(f'{mcm_dir}', f'pseudo2datavec_{spec_name}.npy'), pseudo2datavec)

    plt.figure(figsize=(10, 8))
    plt.imshow(np.log(np.abs(pseudo2datavec)), aspect=100)
    plt.colorbar()
    plt.title(f'pseudo2datavec {spec_name}')
    plt.savefig(opj(f'{plot_dir}', f'pseudo2datavec_{spec_name}'), bbox_inches='tight')
    plt.close()
