'''
This script saves in the `best_fits` directory the following files:
1. `signal_matrix.npy` contains an array (i,j,k) indexed by (field,pol) × (field,pol) × ℓ
2. `signal_matrix_labels.npy` contains a dictionary of (field_pol × field_pol) → (i,j) 
'''
import matplotlib

matplotlib.use('Agg')
import sys, os

import numpy as np
from pspy import pspy_utils, so_dict, so_spectra, so_map_preprocessing
from pspipe_utils import log, misc

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# first let's get a list of all frequency we plan to study
surveys = d['surveys']
lmax = d['lmax']
type = d['type']
spectra = ['TT', 'TE', 'TB', 'ET', 'BT', 'EE', 'EB', 'BE', 'BB']

bestfit_dir = d['best_fits_dir']
couplings_dir = d['couplings_dir']

cosmo_params = d['cosmo_params']

apply_kspace_filter = d["apply_kspace_filter"]

# compatibility with data_analysis, should be industrialized #FIXME
def get_arrays_list(d):
    surveys = d['surveys']
    arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}
    sv_list, ar_list = [], []
    for sv1 in surveys:
        for ar1 in arrays[sv1]:
            for chan1 in arrays[sv1][ar1]:
                sv_list.append(sv1)
                ar_list.append(f'{ar1}_{chan1}')
    return len(sv_list), sv_list, ar_list

narrays, sv_list, ar_list = get_arrays_list(d)

def cmb_matrix_from_file(f_name, _lmax, spectra, input_type='Dl'):
    ps_mat = np.zeros((3, 3, _lmax+1))
    
    l, ps_theory = so_spectra.read_ps(f_name, spectra=spectra)
    assert l[0] == 2, 'the file is expected to start at l=2'
    lmax = min(_lmax, int(max(l)))  # make sure lmax doesn't exceed model lmax
    
    for p1, pol1 in enumerate('TEB'):
        for p2, pol2 in enumerate('TEB'):
            if input_type == 'Dl':
                ps_theory[pol1 + pol2] *= 2 * np.pi / (l * (l + 1))
            ps_mat[p1, p2, 2:(lmax+1)] = ps_theory[pol1 + pol2][:(lmax+1) - 2]
    
    ps_mat[..., lmax+1:] = ps_mat[..., lmax][..., None] # extend with last val
    
    return ps_mat

def foreground_matrix_from_files(f_name_tmp, arrays_list, _lmax, spectra, input_type='Dl'):
    narrays = len(arrays_list)
    fg_mat = np.zeros((narrays, 3, narrays, 3, _lmax+1))
    
    for a1, array1 in enumerate(arrays_list):
        for a2, array2 in enumerate(arrays_list):
            l, fg_theory = so_spectra.read_ps(f_name_tmp.format(array1, array2), spectra=spectra)
            assert l[0] == 2, 'the file is expected to start at l=2'
            lmax = min(_lmax, int(max(l)))  # make sure lmax doesn't exceed model lmax
            
            for p1, pol1 in enumerate('TEB'):
                for p2, pol2 in enumerate('TEB'):
                    if input_type == 'Dl':
                        fg_theory[pol1 + pol2] *=  2 * np.pi / (l * (l + 1))
                    fg_mat[a1, p1, a2, p2, 2:(lmax+1)] = fg_theory[pol1 + pol2][:(lmax+1) - 2]

    fg_mat[..., lmax+1:] = fg_mat[..., lmax][..., None] # extend with last val
    
    return fg_mat

ells = np.arange(lmax+1)

# ps_mat, fg_mat starts from zero, is C_ell as is convention
f_name_cmb = bestfit_dir + '/cmb.dat'
ps_mat = cmb_matrix_from_file(f_name_cmb, lmax, spectra)

f_name_fg = bestfit_dir + '/fg_{}x{}.dat'
array_list = [f'{sv}_{ar}' for sv, ar in zip(sv_list, ar_list)]
fg_mat = foreground_matrix_from_files(f_name_fg, array_list, lmax, spectra) 

# apply beam and transfer function
beamed_signal_model = np.zeros_like(fg_mat)
tfed_beamed_signal_model = np.zeros_like(fg_mat)
for a1, array1 in enumerate(array_list):
    for a2, array2 in enumerate(array_list):
        l1, bl1 = misc.read_beams(d[f'beam_T_{array1}'], d[f'beam_pol_{array1}'])  # diag TEB, l
        l2, bl2 = misc.read_beams(d[f'beam_T_{array2}'], d[f'beam_pol_{array2}'])  # diag TEB, l
        assert np.all(l1[:(lmax+1)] == ells), 'ell of beam from array1 != ells'
        assert np.all(l2[:(lmax+1)] == ells), 'ell of beam from array2 != ells'

        for p1, pol1 in enumerate('TEB'):
            for p2, pol2 in enumerate('TEB'):
                key = f'{array1}_{pol1}', f'{array2}_{pol2}'
                bl = bl1[pol1][:(lmax+1)] * bl2[pol2][:(lmax+1)] # beams defined at beam level
                
                if apply_kspace_filter:
                    sv1, sv2 = sv_list[a1], sv_list[a2]
                    
                    filter_dict1 = d[f"k_filter_{sv1}"]
                    assert filter_dict1['type'] == 'binary_cross', \
                        f'if {sv1=} kfilt, must be binary cross'
                    tf1 = so_map_preprocessing.analytical_tf_vkhk(filter_dict1['vk_mask'], filter_dict1['hk_mask'], ells)

                    filter_dict2 = d[f"k_filter_{sv2}"]
                    assert filter_dict1['type'] == 'binary_cross', \
                        f'if {sv2=} kfilt, must be binary cross'
                    tf2 = so_map_preprocessing.analytical_tf_vkhk(filter_dict2['vk_mask'], filter_dict2['hk_mask'], ells)

                    tf = np.sqrt(tf1 * tf2) # tf defined at ps level
                else:
                    tf = 1
                
                beamed_signal_model[a1, :, a2, :] = bl * (ps_mat + fg_mat[a1, :, a2, :])
                tfed_beamed_signal_model[a1, :, a2, :] = tf * beamed_signal_model[a1, :, a2, :]

np.save(f'{bestfit_dir}/beamed_signal_model.npy', beamed_signal_model)
np.save(f'{bestfit_dir}/tfed_beamed_signal_model.npy', tfed_beamed_signal_model)

# Next, we loop over each of them and apply the appropriate MCM
pseudo_beamed_signal_model = np.zeros_like(beamed_signal_model)
pseudo_tfed_beamed_signal_model = np.zeros_like(tfed_beamed_signal_model)
single_coupling_pols = {'TT': '00', 'TE': '02', 'ET': '02', 'TB': '02', 'BT': '02'}

for a1, array1 in enumerate(array_list):
    for a2, array2 in enumerate(array_list):
        arrs = f'w_{array1}xw_{array2}' if a1 <= a2 else f'w_{array2}xw_{array1}' # canonical

        for inp, out in ((beamed_signal_model, pseudo_beamed_signal_model), (tfed_beamed_signal_model, pseudo_tfed_beamed_signal_model)):
            # handle single coupling polarization combos
            for P1P2 in single_coupling_pols:
                spin = single_coupling_pols[P1P2]
                mcm_file = f'{couplings_dir}/{arrs}_{spin}_coupling.npy'
                M = np.load(mcm_file) * (2*ells + 1) # 2l + 1 across rows
                pol1, pol2 = P1P2
                p1, p2 = 'TEB'.index(pol1), 'TEB'.index(pol2)
                
                out[a1, p1, a2, p2] = M @ inp[a1, p1, a2, p2]

            # read 22 couplings
            Mpp = np.load( f'{couplings_dir}/{arrs}_pp_coupling.npy') * (2*ells + 1) # 2l + 1 across rows
            Mmm = np.load( f'{couplings_dir}/{arrs}_mm_coupling.npy') * (2*ells + 1) # 2l + 1 across rows

            # handle EE BB
            M = np.block([[Mpp, Mmm], [Mmm, Mpp]])
            clee = inp[a1, 1, a2, 1]
            clbb = inp[a1, 2, a2, 2]
            pcl = M @ np.hstack([clee, clbb])
            out[a1, 1, a2, 1] = pcl[:len(clee)]
            out[a1, 2, a2, 2] = pcl[len(clee):]
            
            # handle EB BE
            M = np.block([[Mpp, -Mmm], [-Mmm, Mpp]])
            cleb = inp[a1, 1, a2, 2]
            clbe = inp[a1, 2, a2, 1]
            pcl = M @ np.hstack([cleb, clbe])
            out[a1, 1, a2, 2] = pcl[:len(cleb)]
            out[a1, 2, a2, 1] = pcl[len(cleb):]


np.save(f'{bestfit_dir}/pseudo_beamed_signal_model.npy', pseudo_beamed_signal_model)
np.save(f'{bestfit_dir}/pseudo_tfed_beamed_signal_model.npy', pseudo_beamed_signal_model)