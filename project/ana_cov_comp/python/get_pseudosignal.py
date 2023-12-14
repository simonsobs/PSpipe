"""
This script saves in the `best_fits` directory the following files:
1. `signal_matrix.npy` contains an array (i,j,k) indexed by (field,pol) × (field,pol) × ℓ
2. `signal_matrix_labels.npy` contains a dictionary of (field_pol × field_pol) → (i,j) 
"""
import matplotlib

matplotlib.use("Agg")
import sys, os

import numpy as np
import pylab as plt
from pspipe_utils import kspace, log, simulation, transfer_function
from pspy import pspy_utils, so_dict
from pspipe_utils import log, misc

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# first let's get a list of all frequency we plan to study
surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# let's create the directories to write best fit to disk and for plotting purpose
bestfit_dir = d["best_fits_dir"]
spectra_dir = d["spectra_dir"]
couplings_dir = d['couplings_dir']
plot_dir = d["plot_dir"] + "/best_fits"
binning_file = d["binning_file"]
bin_low, bin_high, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax=lmax+1)

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)

cosmo_params = d["cosmo_params"]

# compatibility with data_analysis, should be industrialized #FIXME
def get_arrays_list(d):
    surveys = d['surveys']
    arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}
    sv_list, ar_list = [], []
    for sv1 in surveys:
        for ar1 in arrays[sv1]:
            for chan1 in arrays[sv1][ar1]:
                sv_list.append(sv1)
                ar_list.append(f"{ar1}_{chan1}")
    return len(sv_list), sv_list, ar_list

narrays, sv_list, ar_list = get_arrays_list(d)


f_name_cmb = bestfit_dir + "/cmb.dat"
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

f_name_fg = bestfit_dir + "/fg_{}x{}.dat"
array_list = [f"{sv}_{ar}" for sv, ar in zip(sv_list, ar_list)]

# fg_mat starts from zero, is C_ell as is convention
_, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, array_list, lmax+1, spectra) 
np.save(f"{bestfit_dir}/signal_matrix.npy", fg_mat)

narrays = len(array_list)
fl_array = np.zeros((3 * narrays, 3 * narrays, lmax))

# for convenience, we provide a dict that maps (field) x (field) to (i,j) in the matrix
labels = {}
signal_cl = {}
for c1, array1 in enumerate(array_list):
    for c2, array2 in enumerate(array_list):
        for s1, pol1 in enumerate("TEB"):
            for s2, pol2 in enumerate("TEB"):
                key = f'{array1}_{pol1}', f'{array2}_{pol2}'
                labels[key] = (c1 + narrays * s1, c2 + narrays * s2)
                signal_cl[key] = fg_mat[c1 + narrays * s1, c2 + narrays * s2, :]
np.save(f"{bestfit_dir}/signal_matrix_labels.npy", labels)


# apply beam and transfer function
beamed_signal_cl = {}
for c1, array1 in enumerate(array_list):
    for c2, array2 in enumerate(array_list):
        l, bl1 = misc.read_beams(d[f"beam_T_{array1}"], d[f"beam_pol_{array1}"])  # diag TEB
        l, bl2 = misc.read_beams(d[f"beam_T_{array2}"], d[f"beam_pol_{array2}"])  # diag TEB
        arrs = f'{array1}x{array2}' if c1 <= c2 else f'{array2}x{array1}'  # canonical
        one_d_tf = np.loadtxt(f"{spectra_dir}/one_dimension_kspace_tf_{arrs}.dat")

        for s1, pol1 in enumerate("TEB"):
            for s2, pol2 in enumerate("TEB"):
                key = f'{array1}_{pol1}', f'{array2}_{pol2}'
                bl = np.sqrt(bl1[pol1][:(lmax+1)] * bl2[pol2][:(lmax+1)])
                tf_start = spectra.index(pol1 + pol2)  # tf is TT,TE,ET,... but binned
                tf = np.interp(np.arange(lmax+1), lb, one_d_tf[tf_start:(tf_start+len(lb))])
                beamed_signal_cl[key] = bl * tf * signal_cl[key]
np.save(f"{bestfit_dir}/beamed_signal_matrix.npy", labels)

# Next, we loop over each of them and apply the appropriate MCM
pseudo_signal_cl = {}
single_coupling_pols = {'TT': '00', 'TE': '02', 'ET': '02', 'TB': '02', 'BT': '02'}

for c1, array1 in enumerate(array_list):
    for c2, array2 in enumerate(array_list):
        arrs = f'w_{array1}xw_{array2}' if c1 <= c2 else f'w_{array2}xw_{array1}'

        # handle single coupling polarization combos
        for P1P2 in single_coupling_pols:
            spin = single_coupling_pols[P1P2]
            mcm_file = f'{couplings_dir}/{arrs}_{spin}_coupling.npy'
            M = np.load(mcm_file) * (2*np.arange(lmax+1)+1)
            pol1, pol2 = P1P2
            cl = beamed_signal_cl[(f'{array1}_{pol1}', f'{array2}_{pol2}')]
            pseudo_signal_cl[(f'{array1}_{pol1}', f'{array2}_{pol2}')] = M @ cl

        # read 22 couplings
        Mpp = np.load( f'{couplings_dir}/{arrs}_pp_coupling.npy') * (2*np.arange(lmax+1)+1)
        Mmm = np.load( f'{couplings_dir}/{arrs}_mm_coupling.npy') * (2*np.arange(lmax+1)+1)

        # handle EE BB
        M = np.block([[Mpp, Mmm], [Mmm, Mpp]])
        clee = beamed_signal_cl[(f'{array1}_E', f'{array2}_E')]
        clbb = beamed_signal_cl[(f'{array1}_B', f'{array2}_B')]
        pcl = M @ np.hstack([clee, clbb])
        pseudo_signal_cl[(f'{array1}_E', f'{array2}_E')] = pcl[:len(clee)]
        pseudo_signal_cl[(f'{array1}_B', f'{array2}_B')] = pcl[len(clee):]
        
        # handle EB BE
        M = np.block([[Mpp, -Mmm], [-Mmm, Mpp]])
        cleb = beamed_signal_cl[(f'{array1}_E', f'{array2}_B')]
        clbe = beamed_signal_cl[(f'{array1}_B', f'{array2}_E')]
        pcl = M @ np.hstack([cleb, clbe])
        pseudo_signal_cl[(f'{array1}_E', f'{array2}_B')] = pcl[:len(cleb)]
        pseudo_signal_cl[(f'{array1}_B', f'{array2}_E')] = pcl[len(cleb):]


np.save(f"{bestfit_dir}/pseudo_signal_matrix.npy", pseudo_signal_cl)  # save a dict
