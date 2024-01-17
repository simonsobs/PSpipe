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
filters_dir = d['filters_dir']

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
log.info("Getting beamed and tf'ed signal")
beamed_signal_model = np.zeros_like(fg_mat)
tfed_beamed_signal_model = np.zeros_like(fg_mat)
for a1, array1 in enumerate(array_list):
    for a2, array2 in enumerate(array_list):
        l1, bl1 = misc.read_beams(d[f'beam_T_{array1}'], d[f'beam_pol_{array1}'])  # diag TEB, l
        l2, bl2 = misc.read_beams(d[f'beam_T_{array2}'], d[f'beam_pol_{array2}'])  # diag TEB, l
        assert np.all(l1[:(lmax+1)] == ells), 'ell of beam from array1 != ells'
        assert np.all(l2[:(lmax+1)] == ells), 'ell of beam from array2 != ells'

        if apply_kspace_filter:
            sv1, sv2 = sv_list[a1], sv_list[a2]
            fl4_1 = np.load(f'{filters_dir}/{sv1}_fl4_fullsky.npy')
            fl4_2 = np.load(f'{filters_dir}/{sv2}_fl4_fullsky.npy')

        for p1, pol1 in enumerate('TEB'):
            for p2, pol2 in enumerate('TEB'):
                bl = bl1[pol1][:(lmax+1)] * bl2[pol2][:(lmax+1)] # beams defined at field level
                
                if apply_kspace_filter:
                    polstr1 = 'T' if pol1 == 'T' else 'pol'
                    rd1 = np.load(f'{filters_dir}/{sv1}_{polstr1}_res_dict.npy', allow_pickle=True).item()
                    tf1 = fl4_1 ** (rd1['binned_spec_cov_diag_alpha4']/2)
                    
                    polstr2 = 'T' if pol2 == 'T' else 'pol'
                    rd2 = np.load(f'{filters_dir}/{sv2}_{polstr2}_res_dict.npy', allow_pickle=True).item()
                    tf2 = fl4_2 ** (rd2['binned_spec_cov_diag_alpha4']/2)

                    tf = np.sqrt(tf1 * tf2) # tf defined at ps level
                else:
                    tf = 1
                
                beamed_signal_model[a1, p1, a2, p2] = bl * (ps_mat[p1, p2] + fg_mat[a1, p1, a2, p2])
                tfed_beamed_signal_model[a1, p1, a2, p2] = tf * beamed_signal_model[a1, p1, a2, p2]

np.save(f'{bestfit_dir}/beamed_signal_model.npy', beamed_signal_model)
np.save(f'{bestfit_dir}/tfed_beamed_signal_model.npy', tfed_beamed_signal_model)

# Next, we loop over each of them and apply the appropriate MCM
log.info("Getting pseudo beamed and tf'ed signal")

pseudo_beamed_signal_model = np.zeros_like(beamed_signal_model)
pseudo_tfed_beamed_signal_model = np.zeros_like(tfed_beamed_signal_model)
single_coupling_pols = {'TT': '00', 'TE': '02', 'ET': '02', 'TB': '02', 'BT': '02'}

for a1, array1 in enumerate(array_list):
    for a2, array2 in enumerate(array_list):
        log.info(f"{array1}x{array2}")
        arrs = f'w_{array1}xw_{array2}' if a1 <= a2 else f'w_{array2}xw_{array1}' # canonical

        # handle single coupling polarization combos
        for P1P2, spin in single_coupling_pols.items():
            M = np.load(f'{couplings_dir}/{arrs}_{spin}_mcm.npy')
            pol1, pol2 = P1P2
            p1, p2 = 'TEB'.index(pol1), 'TEB'.index(pol2)
            for inp, out in ((beamed_signal_model, pseudo_beamed_signal_model), (tfed_beamed_signal_model, pseudo_tfed_beamed_signal_model)):
                out[a1, p1, a2, p2] = M @ inp[a1, p1, a2, p2]

        # handle EE BB
        M = np.load(f'{couplings_dir}/{arrs}_diag_mcm.npy')
        for inp, out in ((beamed_signal_model, pseudo_beamed_signal_model), (tfed_beamed_signal_model, pseudo_tfed_beamed_signal_model)):
            clee = inp[a1, 1, a2, 1]
            clbb = inp[a1, 2, a2, 2]
            pcl = M @ np.hstack([clee, clbb])
            out[a1, 1, a2, 1] = pcl[:len(clee)]
            out[a1, 2, a2, 2] = pcl[len(clee):]
        
        # handle EB BE
        M = np.load(f'{couplings_dir}/{arrs}_off_mcm.npy')
        for inp, out in ((beamed_signal_model, pseudo_beamed_signal_model), (tfed_beamed_signal_model, pseudo_tfed_beamed_signal_model)):
            cleb = inp[a1, 1, a2, 2]
            clbe = inp[a1, 2, a2, 1]
            pcl = M @ np.hstack([cleb, clbe])
            out[a1, 1, a2, 2] = pcl[:len(cleb)]
            out[a1, 2, a2, 1] = pcl[len(cleb):]


np.save(f'{bestfit_dir}/pseudo_beamed_signal_model.npy', pseudo_beamed_signal_model)
np.save(f'{bestfit_dir}/pseudo_tfed_beamed_signal_model.npy', pseudo_tfed_beamed_signal_model)