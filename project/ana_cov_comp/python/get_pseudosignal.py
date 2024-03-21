description = """
This script computes the effective signal pseudospectra, which is needed for
the covariance matrix under the INKA approximation. Because the covariance has a
different effective kspace tf applied to the power spectra, we need to apply
that 4pt tf to the signal power spectra from get_best_fit_mflike.py, and then 
couple that quantity using the mode coupling matrix of the analysis mask.
Finally, also in accordance with INKA, we divide by the relevant w2 factors. 
Importantly, we assume the signal model (including, e.g., the beam) does
not depend on the split.

It is short enough that it should always run in a one-shot job, so it 
accepts no arguments other than paramfile.
"""
import matplotlib

matplotlib.use('Agg')

import numpy as np
from pspy import so_dict
from pspipe_utils import log, misc, pspipe_list, covariance as psc
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

# first let's get a list of all frequency we plan to study
surveys = d['surveys']
lmax_pseudocov = d['lmax_pseudocov']
assert lmax_pseudocov >= d['lmax'], \
    f"{lmax_pseudocov=} must be >= {d['lmax']=}" 
type = d['type']
spectra = ['TT', 'TE', 'TB', 'ET', 'BT', 'EE', 'EB', 'BE', 'BB']

bestfit_dir = d['best_fits_dir']
couplings_dir = d['couplings_dir']
ewin_alms_dir = d['ewin_alms_dir']
filters_dir = d['filters_dir']

cosmo_params = d['cosmo_params']

apply_kspace_filter = d["apply_kspace_filter"]

# format:
# - unroll all 'fields' i.e. (survey x array x chan x split x pol) is a 'field'
# - any given combination is then ('field' x 'field')
#
# notes:
# - we are 'hardcoding' that all splits for a given field have the same
# analysis mask, and that all pols for a given field have the same 
# sigma map.

# we define the canon by the windows order. we first build the fields,
# then use a mapping from fields to windows to build the canonical
# windows
sv_ar_chans = [] # necessary for indexing signal model
field_infos = []
ewin_infos = []
for sv1 in sv2arrs2chans:
    for ar1 in sv2arrs2chans[sv1]:
        for chan1 in sv2arrs2chans[sv1][ar1]:
            sv_ar_chans.append((sv1, ar1, chan1)) 
            for split1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
                for pol1 in ['T', 'P']:
                    field_info = (sv1, ar1, chan1, split1, pol1)
                    if field_info not in field_infos:
                        field_infos.append(field_info)
                    else:
                        raise ValueError(f'{field_info=} is not unique')
                    
                    ewin_info_s = psc.get_ewin_info_from_field_info(field_info, d, mode='w', return_paths_ops=True)
                    if ewin_info_s not in ewin_infos:
                        ewin_infos.append(ewin_info_s)
                    else:
                        pass

                    ewin_info_n = psc.get_ewin_info_from_field_info(field_info, d, mode='ws', extra='sqrt_pixar', return_paths_ops=True)
                    if ewin_info_n not in ewin_infos:
                        ewin_infos.append(ewin_info_n)
                    else:
                        pass

ells = np.arange(lmax_pseudocov+1)

# ps_mat, fg_mat starts from zero, is C_ell as is convention
f_name_cmb = bestfit_dir + '/cmb.dat'
ps_mat = psc.cmb_matrix_from_file(f_name_cmb, lmax_pseudocov, spectra)

f_name_fg = bestfit_dir + '/fg_{}_{}_{}x{}_{}_{}.dat'
fg_mat = psc.foreground_matrix_from_files(f_name_fg, sv_ar_chans, lmax_pseudocov, spectra)

# apply beam and transfer function
log.info("Getting beamed and tf'ed signal")
beamed_signal_model = np.zeros_like(fg_mat)
tfed_beamed_signal_model = np.zeros_like(fg_mat)
for sac1, sv_ar_chan1 in enumerate(sv_ar_chans):
    sv1, ar1, chan1 = sv_ar_chan1
    for sac2, sv_ar_chan2 in enumerate(sv_ar_chans):
        sv2, ar2, chan2 = sv_ar_chan2

        l1, bl1 = misc.read_beams(d[f'beam_T_{sv1}_{ar1}_{chan1}'], d[f'beam_pol_{sv1}_{ar1}_{chan1}'])  # diag TEB, l
        l2, bl2 = misc.read_beams(d[f'beam_T_{sv2}_{ar2}_{chan2}'], d[f'beam_pol_{sv2}_{ar2}_{chan2}'])  # diag TEB, l
        assert np.all(l1[:(lmax_pseudocov+1)] == ells), 'ell of beam from array1 != ells'
        assert np.all(l2[:(lmax_pseudocov+1)] == ells), 'ell of beam from array2 != ells'

        if apply_kspace_filter:
            # fl4_1 = np.load(f'{filters_dir}/{sv1}_fl_4pt_fullsky.npy')
            # fl4_2 = np.load(f'{filters_dir}/{sv2}_fl_4pt_fullsky.npy')
            fl2_1 = np.load(f'{filters_dir}/{sv1}_fl_2pt_fullsky.npy')
            fl2_2 = np.load(f'{filters_dir}/{sv2}_fl_2pt_fullsky.npy')

        for p1, pol1 in enumerate('TEB'):
            for p2, pol2 in enumerate('TEB'):
                bl = bl1[pol1][:(lmax_pseudocov+1)] * bl2[pol2][:(lmax_pseudocov+1)] # beams defined at field level
                
                if apply_kspace_filter:
                    polstr1 = 'T' if pol1 == 'T' else 'pol'
                    rd1 = np.load(f'{filters_dir}/{sv1}_{polstr1}_res_dict.npy', allow_pickle=True).item()
                    tf1 = fl2_1 ** (rd1['binned_power_cov_alpha']/2)
                    
                    polstr2 = 'T' if pol2 == 'T' else 'pol'
                    rd2 = np.load(f'{filters_dir}/{sv2}_{polstr2}_res_dict.npy', allow_pickle=True).item()
                    tf2 = fl2_2 ** (rd2['binned_power_cov_alpha']/2)

                    tf = np.sqrt(tf1 * tf2) # tf defined at ps level
                else:
                    tf = 1
                
                beamed_signal_model[sac1, p1, sac2, p2] = bl * (ps_mat[p1, p2] + fg_mat[sac1, p1, sac2, p2])
                tfed_beamed_signal_model[sac1, p1, sac2, p2] = tf * beamed_signal_model[sac1, p1, sac2, p2]

np.save(f'{bestfit_dir}/beamed_signal_model.npy', beamed_signal_model)
np.save(f'{bestfit_dir}/tfed_beamed_signal_model.npy', tfed_beamed_signal_model)

# Next, we loop over each of them and apply the appropriate MCM
log.info("Getting pseudo beamed and tf'ed signal")

pseudo_beamed_signal_model = np.zeros_like(beamed_signal_model)
pseudo_tfed_beamed_signal_model = np.zeros_like(tfed_beamed_signal_model)

single_coupling_pols = {'TT': '00', 'TE': '02', 'ET': '02', 'TB': '02', 'BT': '02'}

for sac1, sv_ar_chan1 in enumerate(sv_ar_chans):
    sv1, ar1, chan1 = sv_ar_chan1

    # check that for each pol combo, all splits give the 
    # same effective windows for the signal (as stated above,
    # we assume the signal windows are independent of split)
    for TP1 in ('T', 'P'):
        for s1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
            field_info1 = (sv1, ar1, chan1, s1, TP1)
            if s1 == 0:
                ewin_info1 = psc.get_ewin_info_from_field_info(field_info1, d, mode='w')
            else:
                _ewin_info1 = psc.get_ewin_info_from_field_info(field_info1, d, mode='w')
                assert _ewin_info1 == ewin_info1, \
                    f'{_ewin_info1=} for split {s1=} of {sv1=}, {ar1=}, {chan1=}, {TP1=} ' + \
                    f'does not equal {ewin_info1=} for split s1=0'

    for sac2, sv_ar_chan2 in enumerate(sv_ar_chans):
        sv2, ar2, chan2 = sv_ar_chan2

        # handle single coupling polarization combos
        for P1P2, spin in single_coupling_pols.items():
            # get canonical inputs
            pol1, pol2 = P1P2
            TP1, p1 = psc.pol2pol_info(pol1)
            TP2, p2 = psc.pol2pol_info(pol2)
            field_info1 = (sv1, ar1, chan1, 0, TP1) # split=0 since we tested does not depend on split
            field_info2 = (sv2, ar2, chan2, 0, TP2) # spl is fixed, everything else iterates
            ewin_name1, ewin_name2 = psc.canonize_connected_2pt(
                psc.get_ewin_info_from_field_info(field_info1, d, mode='w'),
                psc.get_ewin_info_from_field_info(field_info2, d, mode='w'),
                ewin_infos
                ) 
            log.info(f'Convolving and dividing by w2: {ewin_name1}, {ewin_name2}')

            M = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{spin}_mcm.npy')
            w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
            
            for inp, out in ((beamed_signal_model, pseudo_beamed_signal_model), (tfed_beamed_signal_model, pseudo_tfed_beamed_signal_model)):
                out[sac1, p1, sac2, p2] = M @ inp[sac1, p1, sac2, p2] / w2
                    
        # handle quad-coupling polarization combos
        field_info1 = (sv1, ar1, chan1, 0, 'P')
        field_info2 = (sv2, ar2, chan2, 0, 'P')
        ewin_name1, ewin_name2 = psc.canonize_connected_2pt(
            psc.get_ewin_info_from_field_info(field_info1, d, mode='w'),
            psc.get_ewin_info_from_field_info(field_info2, d, mode='w'),
            ewin_infos
            ) 
        log.info(f'Convolving and dividing by w2: {ewin_name1}, {ewin_name2}')
        
        w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')

        # handle EE BB
        M = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_diag_mcm.npy')
        
        for inp, out in ((beamed_signal_model, pseudo_beamed_signal_model), (tfed_beamed_signal_model, pseudo_tfed_beamed_signal_model)):
            clee = inp[sac1, 1, sac2, 1]
            clbb = inp[sac1, 2, sac2, 2]
            pcl = M @ np.hstack([clee, clbb]) / w2
            out[sac1, 1, sac2, 1] = pcl[:len(clee)]
            out[sac1, 2, sac2, 2] = pcl[len(clee):]
        
        # handle EB BE
        M = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_off_mcm.npy')
        
        for inp, out in ((beamed_signal_model, pseudo_beamed_signal_model), (tfed_beamed_signal_model, pseudo_tfed_beamed_signal_model)):
            cleb = inp[sac1, 1, sac2, 2]
            clbe = inp[sac1, 2, sac2, 1]
            pcl = M @ np.hstack([cleb, clbe]) / w2
            out[sac1, 1, sac2, 2] = pcl[:len(cleb)]
            out[sac1, 2, sac2, 1] = pcl[len(cleb):]

np.save(f'{bestfit_dir}/pseudo_beamed_signal_model.npy', pseudo_beamed_signal_model)
np.save(f'{bestfit_dir}/pseudo_tfed_beamed_signal_model.npy', pseudo_tfed_beamed_signal_model)