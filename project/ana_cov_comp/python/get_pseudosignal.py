'''
This script computes the effective signal pseudospectra, which is needed for
the covariance matrix under the INKA approximation. Because the covariance has a
different effective kspace tf applied to the power spectra, we need to apply
that 4pt tf to the signal power spectra from get_best_fit_mflike.py, and then 
couple that quantity using the mode coupling matrix of the analysis mask.
Finally, also in accordance with INKA, we divide by the relevant w2 factors. 
'''
import matplotlib

matplotlib.use('Agg')
import sys

import numpy as np
from pspy import so_dict
from pspipe_utils import log, misc, pspipe_list, covariance as psc

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

narrays, sv_list, ar_list = pspipe_list.get_arrays_list(d)

ells = np.arange(lmax+1)

# ps_mat, fg_mat starts from zero, is C_ell as is convention
f_name_cmb = bestfit_dir + '/cmb.dat'
ps_mat = psc.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

f_name_fg = bestfit_dir + '/fg_{}x{}.dat'
array_list = [f'{sv}_{ar}' for sv, ar in zip(sv_list, ar_list)]
fg_mat = psc.foreground_matrix_from_files(f_name_fg, array_list, lmax, spectra)

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
            fl4_1 = np.load(f'{filters_dir}/{sv1}_fl_4pt_fullsky.npy')
            fl4_2 = np.load(f'{filters_dir}/{sv1}_fl_4pt_fullsky.npy')

        for p1, pol1 in enumerate('TEB'):
            for p2, pol2 in enumerate('TEB'):
                bl = bl1[pol1][:(lmax+1)] * bl2[pol2][:(lmax+1)] # beams defined at field level
                
                if apply_kspace_filter:
                    polstr1 = 'T' if pol1 == 'T' else 'pol'
                    rd1 = np.load(f'{filters_dir}/{sv1}_{polstr1}_res_dict.npy', allow_pickle=True).item()
                    tf1 = fl4_1 ** (rd1['binned_power_cov_alpha']/2)
                    
                    polstr2 = 'T' if pol2 == 'T' else 'pol'
                    rd2 = np.load(f'{filters_dir}/{sv2}_{polstr2}_res_dict.npy', allow_pickle=True).item()
                    tf2 = fl4_2 ** (rd2['binned_power_cov_alpha']/2)

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
        # FIXME: replace with actual ewin/canonization
        arrs = f'w_{array1}xw_{array2}' if a1 <= a2 else f'w_{array2}xw_{array1}' # canonical

        log.info(f'Convolving {arrs} and dividing by w2')

        w2 = np.load(f'{couplings_dir}/{arrs}_w2.npy')

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

        pseudo_beamed_signal_model[a1, :, a2, :] /= w2
        pseudo_tfed_beamed_signal_model[a1, :, a2, :] /= w2

np.save(f'{bestfit_dir}/pseudo_beamed_signal_model.npy', pseudo_beamed_signal_model)
np.save(f'{bestfit_dir}/pseudo_tfed_beamed_signal_model.npy', pseudo_tfed_beamed_signal_model)