'''
This script computes the effective noise pseudospectra, which is needed for
the covariance matrix under the INKA approximation. Because the covariance has a
different effective kspace tf applied to the power spectra, we need to apply
that 4pt tf to the noise power spectra from get_noise_model.py, and then 
couple that quantity using the mode coupling matrix of the noise mask.
The effect of get_noise_model together with get_pseudo_noise is to basically
"replace" the 2pt kspace tf in the measured noise pseudospectra with the 
4pt tf. Finally, also in accordance with INKA, we divide by the relevant w2
factors. 
'''
import matplotlib

matplotlib.use('Agg')
import sys

import numpy as np
from pspy import so_dict
from pspipe_utils import log, pspipe_list, covariance as psc

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

noise_model_dir = d['noise_model_dir']
couplings_dir = d['couplings_dir']
ewin_alms_dir = d['ewin_alms_dir']
filters_dir = d['filters_dir']

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
field_infos = []
ewin_infos = []
for sv1 in sv2arrs2chans:
    for ar1 in sv2arrs2chans[sv1]:
        for chan1 in sv2arrs2chans[sv1][ar1]:
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

single_coupling_pols = {'TT': '00', 'TE': '02', 'ET': '02', 'TB': '02', 'BT': '02'}

# we will make noise models for each survey and array,
# so "everything" happens inside this loop
for sv1 in sv2arrs2chans:
    for ar1 in sv2arrs2chans[sv1]:

        for i, chan1 in enumerate(sv2arrs2chans[sv1][ar1]):
            if i == 0:
                nsplit = len(d[f'maps_{sv1}_{ar1}_{chan1}'])
            else:
                _nsplit = len(d[f'maps_{sv1}_{ar1}_{chan1}'])
                assert _nsplit == nsplit, \
                    f'sv={sv1}, ar={ar1}, chan={chan1}, nsplit={_nsplit}, expected {nsplit}'
                
        for split1 in range(nsplit):
            log.info(f'Doing {sv1}, {ar1}, set{split1}')
            log.info("Getting tf'ed noise model")

            # this has shape (nchan, npol, nchan, npol, nell)
            noise_model = np.load(f'{noise_model_dir}/{sv1}_{ar1}_set{split1}_noise_model.npy')

            # apply transfer function
            tfed_noise_model = np.zeros_like(noise_model)

            if apply_kspace_filter:
                # fl4 = np.load(f'{filters_dir}/{sv1}_fl_4pt_fullsky.npy')
                fl2 = np.load(f'{filters_dir}/{sv1}_fl_2pt_fullsky.npy')

            for p1, pol1 in enumerate('TEB'):
                for p2, pol2 in enumerate('TEB'):
                    
                    if apply_kspace_filter:
                        polstr1 = 'T' if pol1 == 'T' else 'pol'
                        rd1 = np.load(f'{filters_dir}/{sv1}_{polstr1}_res_dict.npy', allow_pickle=True).item()
                        tf1 = fl2 ** (rd1['binned_power_cov_alpha']/2)
                        
                        polstr2 = 'T' if pol2 == 'T' else 'pol'
                        rd2 = np.load(f'{filters_dir}/{sv1}_{polstr2}_res_dict.npy', allow_pickle=True).item()
                        tf2 = fl2 ** (rd2['binned_power_cov_alpha']/2)

                        tf = np.sqrt(tf1 * tf2) # tf defined at ps level
                    else:
                        tf = 1
                    
                    tfed_noise_model[:, p1, :, p2] = tf * noise_model[:, p1, :, p2]

            np.save(f'{noise_model_dir}/{sv1}_{ar1}_set{split1}_tfed_noise_model.npy', tfed_noise_model)

            # Next, we loop over each of them and apply the appropriate MCM
            log.info("Getting pseudo tf'ed noise model")

            pseudo_tfed_noise_model = np.zeros_like(tfed_noise_model)

            for c1, chan1 in enumerate(sv2arrs2chans[sv1][ar1]):
                for c2, chan2 in enumerate(sv2arrs2chans[sv1][ar1]):                    

                    # handle single coupling polarization combos
                    for P1P2, spin in single_coupling_pols.items():
                        # get canonical inputs
                        pol1, pol2 = P1P2
                        TP1, p1 = psc.pol2pol_info(pol1)
                        TP2, p2 = psc.pol2pol_info(pol2)
                        field_info1 = (sv1, ar1, chan1, split1, TP1)
                        field_info2 = (sv1, ar1, chan2, split1, TP2) # sv, ar, spl are fixed, ch and p iterate
                        ewin_name1, ewin_name2 = psc.canonize_connected_2pt(
                            psc.get_ewin_info_from_field_info(field_info1, d, mode='ws', extra='sqrt_pixar'),
                            psc.get_ewin_info_from_field_info(field_info2, d, mode='ws', extra='sqrt_pixar'),
                            ewin_infos
                            ) 
                        log.info(f'Convolving and dividing by w2: {ewin_name1}, {ewin_name2}')

                        M = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{spin}_mcm.npy')
                        w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')

                        pseudo_tfed_noise_model[c1, p1, c2, p2] = M @ tfed_noise_model[c1, p1, c2, p2] / w2

                    # handle quad-coupling polarization combos
                    field_info1 = (sv1, ar1, chan1, split1, 'P')
                    field_info2 = (sv1, ar1, chan2, split1, 'P')
                    ewin_name1, ewin_name2 = psc.canonize_connected_2pt(
                        psc.get_ewin_info_from_field_info(field_info1, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_info2, d, mode='ws', extra='sqrt_pixar'),
                        ewin_infos
                        ) 
                    log.info(f'Convolving and dividing by w2: {ewin_name1}, {ewin_name2}')
                    
                    w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
                        
                    # handle EE BB
                    M = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_diag_mcm.npy')
                    
                    clee = tfed_noise_model[c1, 1, c2, 1]
                    clbb = tfed_noise_model[c1, 2, c2, 2]
                    pcl = M @ np.hstack([clee, clbb]) / w2
                    pseudo_tfed_noise_model[c1, 1, c2, 1] = pcl[:len(clee)]
                    pseudo_tfed_noise_model[c1, 2, c2, 2] = pcl[len(clee):]
                    
                    # handle EB BE
                    M = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_off_mcm.npy')

                    cleb = tfed_noise_model[c1, 1, c2, 2]
                    clbe = tfed_noise_model[c1, 2, c2, 1]
                    pcl = M @ np.hstack([cleb, clbe]) / w2
                    pseudo_tfed_noise_model[c1, 1, c2, 2] = pcl[:len(cleb)]
                    pseudo_tfed_noise_model[c1, 2, c2, 1] = pcl[len(cleb):]
            
            np.save(f'{noise_model_dir}/{sv1}_{ar1}_set{split1}_pseudo_tfed_noise_model.npy', pseudo_tfed_noise_model)