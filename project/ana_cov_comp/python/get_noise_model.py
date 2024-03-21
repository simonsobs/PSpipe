description = """
This script computes the noise model for the covariance from the measured alms,
using auto - crosses. We smooth the noisy measurement of the noise pseudospectra
from the data using a Savitzky Golay filter. Because the covariance has a
different effective kspace tf applied to the power spectra, we need to
deconvolve the mask and the 2pt tf from the pseudospectra. Later, we'll reapply
the 4pt tf and mask in get_pseudonoise.py.

It is short enough that it should always run in a one-shot job, so it 
accepts no arguments other than paramfile.
"""
from pspipe_utils import log, pspipe_list, covariance as psc
from pspy import so_dict, pspy_utils

from pixell import curvedsky

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from itertools import combinations, combinations_with_replacement as cwr, product
import os
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

lmax_pseudocov = d['lmax_pseudocov']
assert lmax_pseudocov >= d['lmax'], \
    f"{lmax_pseudocov=} must be >= {d['lmax']=}" 

alms_dir = d['alms_dir']
savgol_w = d['savgol_w']
savgol_k = d['savgol_k']

noise_model_dir = d['noise_model_dir']
couplings_dir = d['couplings_dir']
filters_dir = d['filters_dir']
plot_dir = os.path.join(d['plot_dir'], 'noise_model')
pspy_utils.create_directory(noise_model_dir)
pspy_utils.create_directory(plot_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

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
        log.info(f'Doing {sv1}, {ar1}')

        # load alms for this survey and array
        # we are assuming:
        # - noise is uncorrelated between surveys 
        # - noise is uncorrelated between arrays within surveys
        # - alms have shape (npol=3, nalm)
        alms = []
        nchan = len(sv2arrs2chans[sv1][ar1])
        
        for i, chan1 in enumerate(sv2arrs2chans[sv1][ar1]):
            if i == 0:
                nsplit = len(d[f'maps_{sv1}_{ar1}_{chan1}'])
            else:
                _nsplit = len(d[f'maps_{sv1}_{ar1}_{chan1}'])
                assert _nsplit == nsplit, \
                    f'sv={sv1}, ar={ar1}, chan={chan1}, nsplit={_nsplit}, expected {nsplit}'
            
            for split1 in range(nsplit):
                alms.append(np.load(f'{alms_dir}/alms_pseudocov_{sv1}_{ar1}_{chan1}_{split1}.npy'))
                assert alms[-1].ndim == 2, \
                    f'sv={sv1}, ar={ar1}, chan={chan1}, split={split1} alm.ndim={alms[-1].ndim}, expected 2'
                assert alms[-1].shape[0] == 3, \
                    f'sv={sv1}, ar={ar1}, chan={chan1}, split={split1} alm.shape[0]={alms[-1].shape[0]}, expected 3'
        
        alms = np.asarray(alms).reshape(nchan, nsplit, 3, -1) # will fail if shape mismatches

        # get the signal model for this survey and array from
        # the average of split cross spectra. 
        # we are assuming:
        # - noise is uncorrelated between splits within arrays
        nell = curvedsky.nalm2lmax(alms.shape[-1]) + 1
        assert nell == lmax_pseudocov, \
            f'Expected nell={lmax_pseudocov}, got {nell=}'

        signal_model = 0
        count = 0
        for split1, split2 in combinations(range(nsplit), r=2): # 01 02 03 12 13 23
            alms1, alms2 = alms[:, split1], alms[:, split2]
            
            _signal_model = np.zeros((nchan, 3, nchan, 3, nell), dtype=alms.real.dtype)
            
            # signal model is not guaranteed to be symmetric cause of split crosses
            for preidx1, preidx2 in product(np.ndindex((nchan, 3)), repeat=2):
                _signal_model[(*preidx1, *preidx2)] = curvedsky.alm2cl(alms1[preidx1], alms2[preidx2])
            
            signal_model += _signal_model
            count += 1
        assert count == nsplit * (nsplit - 1) / 2, \
            f'Calculated {count=} but expected {nsplit * (nsplit - 1) / 2}'
        signal_model /= count
        
        # signal model is not guaranteed to be symmetric cause of split crosses,
        # but we know it should be. in effect, we get extra independent samples
        # of the off diagonals by virtue of there being 2 of them. alternatively,
        # could have iterated split1, split2 over permutations in above loop
        signal_model = (signal_model + np.moveaxis(signal_model, (0, 1), (2, 3))) / 2

        # each noise model will have the following shape:
        # (nchan, npol=3, nchan, npol=3, nell)
        # we are assuming:
        # - noise may be correlated between channels and pols
        # - beams and masks don't vary over split
        for split1 in range(nsplit):
            log.info(f'Doing {sv1}, {ar1}, set{split1}')

            alms1 = alms[:, split1]
            total_model = np.zeros((nchan, 3, nchan, 3, nell), dtype=alms.real.dtype)
            
            # total model is guaranteed to be symmetric cause of split auto
            for preidx1, preidx2 in cwr(np.ndindex((nchan, 3)), r=2):
                total_model[(*preidx1, *preidx2)] = curvedsky.alm2cl(alms1[preidx1], alms1[preidx2])
                if preidx1 != preidx2:
                    total_model[(*preidx2, *preidx1)] = total_model[(*preidx1, *preidx2)]

            inp_noise_model = total_model - signal_model

            # huzzah! now we have a noisy thing that we want to smooth, 
            # nonparametrically. we need to fit the diagonals first, because
            # we will fit the off-diagonal corrs, and need to convert them back
            # to cross-spectra using the fit diagonals; i.e., the fit diagonals
            # all need to exist before doing the off-diagonals.
            out_noise_model = np.zeros_like(inp_noise_model)

            inp_corr_model = np.zeros_like(inp_noise_model)
            out_corr_model = np.zeros_like(inp_noise_model)
            
            mask_model = np.zeros_like(inp_noise_model, dtype=bool)

            # autos
            for preidx1 in np.ndindex(nchan, 3):

                # only if we are TT, we start at l=0, otherwise l=2
                if preidx1[1] == 0:
                    start = 0
                else:
                    assert np.all(inp_noise_model[(*preidx1, *preidx1)][:2] == 0), \
                        f'{preidx1=}; expected 0 for l=0,1'
                    start = 2

                y = inp_noise_model[(*preidx1, *preidx1)][start:].copy()

                # might start <= 0, so first we clip and fit without log, and
                # add fitted pts back where it is <= 0. in principle
                # this could still fail, so if it doesn't, then this was a 
                # fluctuation consistent with noise
                mask = y <= 0
                if np.any(mask):
                    log.info(f'{preidx1=} has {np.sum(mask)} values <= 0')
                np.clip(y, 0, None, out=y)
                fit_y = savgol_filter(y, savgol_w, savgol_k)
                y[mask] = fit_y[mask]
                assert not np.any(y <= 0), \
                    log.info(f'{preidx1=} still has {np.sum(y <= 0)} values <= 0 after correcting')
                mask_model[(*preidx1, *preidx1)][start:] = mask

                # now fit in log space
                fit_y = savgol_filter(np.log(y), savgol_w, savgol_k)
                fit_y = np.exp(fit_y)
                out_noise_model[(*preidx1, *preidx1)][start:] = fit_y

            # crosses
            # fit the correlations, not the raw spectra
            # because it's easier to keep the values to physical values (r=-1 to 1)
            for preidx1, preidx2 in combinations(np.ndindex((nchan, 3)), r=2):

                # only if we are TT, we start at l=0, otherwise l=2
                if preidx1[1] == 0 and preidx2[1] == 0:
                    start = 0
                else:
                    assert np.all(inp_noise_model[(*preidx1, *preidx2)][:2] == 0), \
                        f'{preidx1=}, {preidx2=}; expected 0 for l=0,1'
                    start = 2

                y = inp_noise_model[(*preidx1, *preidx2)][start:].copy()

                # normalize by the auto fits instead of the raw autos
                # because the fit is smooth / what we think is "the truth"
                y /= np.sqrt(out_noise_model[(*preidx1, *preidx1)][start:]) 
                y /= np.sqrt(out_noise_model[(*preidx2, *preidx2)][start:])

                # first do fit on corrs, get out_corr_model
                # then turn out_corr_model into out_noise_model
                inp_corr_model[(*preidx1, *preidx2)][start:] = y
                inp_corr_model[(*preidx2, *preidx1)][start:] = y

                # might start >= |1|, so first we clip and fit without arctanh, and
                # add fitted pts back where it is >= |1|. in principle
                # this could still fail, so if it doesn't, then this was a 
                # fluctuation consistent with noise
                mask = np.logical_or(y <= -1, 1 <= y) 
                if np.any(mask):
                    log.info(f'{preidx1=}, {preidx2=} has {np.sum(mask)} values >= |1|')
                np.clip(y, -1, 1, out=y)
                fit_y = savgol_filter(y, savgol_w, savgol_k)
                y[mask] = fit_y[mask]
                assert not np.any(np.logical_or(y <= -1, 1 <= y)), \
                    log.info(f'{preidx1=}, {preidx2=} still has {np.logical_or(y <= -1, 1 <= y)} values >= |1| after correcting')
                mask_model[(*preidx1, *preidx2)][start:] = mask
                mask_model[(*preidx2, *preidx1)][start:] = mask

                # now fit in arctanh space
                fit_y = savgol_filter(np.arctanh(y), savgol_w, savgol_k)
                fit_y = np.tanh(fit_y)
                out_corr_model[(*preidx1, *preidx2)][start:] = fit_y
                out_corr_model[(*preidx2, *preidx1)][start:] = fit_y

                fit_y *= np.sqrt(out_noise_model[(*preidx1, *preidx1)][start:])
                fit_y *= np.sqrt(out_noise_model[(*preidx2, *preidx2)][start:])
                out_noise_model[(*preidx1, *preidx2)][start:] = fit_y
                out_noise_model[(*preidx2, *preidx1)][start:] = fit_y

            np.save(f'{noise_model_dir}/{sv1}_{ar1}_set{split1}_pseudo_spec.npy', out_noise_model)

            out_noise_model_spec = np.zeros_like(out_noise_model)

            # now we need to deconvolve the mask
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
                        log.info(f'Deconvolving: {ewin_name1}, {ewin_name2}')

                        Minv = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{spin}_mcm_inv.npy')

                        out_noise_model_spec[c1, p1, c2, p2] = Minv @ out_noise_model[c1, p1, c2, p2]

                    # handle quad-coupling polarization combos
                    field_info1 = (sv1, ar1, chan1, split1, 'P')
                    field_info2 = (sv1, ar1, chan2, split1, 'P')
                    ewin_name1, ewin_name2 = psc.canonize_connected_2pt(
                        psc.get_ewin_info_from_field_info(field_info1, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_info2, d, mode='ws', extra='sqrt_pixar'),
                        ewin_infos
                        ) 
                    log.info(f'Deconvolving: {ewin_name1}, {ewin_name2}')

                    # handle EE BB
                    Minv = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_diag_mcm_inv.npy')
                    
                    pclee = out_noise_model[c1, 1, c2, 1]
                    pclbb = out_noise_model[c1, 2, c2, 2]
                    cl = Minv @ np.hstack([pclee, pclbb])
                    out_noise_model_spec[c1, 1, c2, 1] = cl[:len(pclee)]
                    out_noise_model_spec[c1, 2, c2, 2] = cl[len(pclee):]
                    
                    # handle EB BE
                    Minv = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_off_mcm_inv.npy')
                    
                    pcleb = out_noise_model[c1, 1, c2, 2]
                    pclbe = out_noise_model[c1, 2, c2, 1]
                    cl = Minv @ np.hstack([pcleb, pclbe])
                    out_noise_model_spec[c1, 1, c2, 2] = cl[:len(pcleb)]
                    out_noise_model_spec[c1, 2, c2, 1] = cl[len(pcleb):]
            
            # now we deconvolve the tf
            if apply_kspace_filter: 
                fl2 = np.load(f'{filters_dir}/{sv1}_fl_2pt_fullsky.npy')

                for p1, pol1 in enumerate('TEB'):
                    for p2, pol2 in enumerate('TEB'):      
                        polstr1 = 'T' if pol1 == 'T' else 'pol'
                        rd1 = np.load(f'{filters_dir}/{sv1}_{polstr1}_res_dict.npy', allow_pickle=True).item()
                        tf1 = fl2 ** rd1['pseudo_spec_alpha']
                        
                        polstr2 = 'T' if pol2 == 'T' else 'pol'
                        rd2 = np.load(f'{filters_dir}/{sv1}_{polstr2}_res_dict.npy', allow_pickle=True).item()
                        tf2 = fl2 ** rd2['pseudo_spec_alpha']

                        tf = np.sqrt(tf1 * tf2) # tf defined at ps level

                        out_noise_model_spec[:, p1, :, p2] = np.divide(
                            out_noise_model_spec[:, p1, :, p2], tf, where=tf!=0, out=np.zeros_like(out_noise_model_spec[:, p1, :, p2])
                            )

            np.save(f'{noise_model_dir}/{sv1}_{ar1}_set{split1}_noise_model.npy', out_noise_model_spec)

            # plot and save
            for preidx1, preidx2 in cwr(np.ndindex((nchan, 3)), r=2):

                if preidx1 == preidx2:
                    # ps
                    raw_y = inp_noise_model[(*preidx1, *preidx2)]
                    fit_y = out_noise_model[(*preidx1, *preidx2)]
                    red_mask = mask_model[(*preidx1, *preidx2)]
                    l = np.arange(raw_y.size)
                    fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row',
                                            figsize=(12, 8), dpi=100, height_ratios=(2, 1),
                                            layout='constrained')
                    axs[0, 0].loglog(l, raw_y, alpha=0.3, label='raw')
                    axs[0, 0].loglog(l, fit_y, alpha=1, label='fit')
                    axs[0, 0].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')
                    axs[0, 0].set_ylabel('$N_{\ell}$', fontsize=12)
                    axs[0, 0].legend()
                    axs[0, 0].grid()

                    axs[1, 0].semilogx(l, (fit_y / raw_y), alpha=1, color='k')
                    axs[1, 0].scatter(l[red_mask], (fit_y / raw_y)[red_mask], alpha=1, c='r')
                    axs[1, 0].set_xlabel('$\ell$', fontsize=12)
                    axs[1, 0].set_ylabel('fit / raw', fontsize=12)
                    axs[1, 0].grid()

                    axs[0, 1].semilogy(l, raw_y, alpha=0.3, label='raw')
                    axs[0, 1].semilogy(l, fit_y, alpha=1, label='fit')
                    axs[0, 1].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')
                    axs[0, 1].legend()
                    axs[0, 1].grid()

                    axs[1, 1].plot(l, (fit_y / raw_y), alpha=1, color='k')
                    axs[1, 1].scatter(l[red_mask], (fit_y / raw_y)[red_mask], alpha=1, c='r')
                    axs[1, 1].set_xlabel('$\ell$', fontsize=12)
                    axs[1, 1].grid()

                    chan1, chan2 = sv2arrs2chans[sv1][ar1][preidx1[0]], sv2arrs2chans[sv1][ar1][preidx2[0]]
                    pol1, pol2 = 'TEB'[preidx1[1]], 'TEB'[preidx2[1]]
                    fig.suptitle(f'{sv1}_{ar1}_set{split1} {chan1}_{pol1}x{chan2}_{pol2}')

                    plt_fn = f'{plot_dir}/{sv1}_{ar1}_set{split1}_{chan1}_{pol1}x{chan2}_{pol2}_ps.png'
                    plt.savefig(plt_fn, bbox_inches='tight')
                    plt.close()

                else:
                    # corrs, ps
                    for i, (raw_y, fit_y) in enumerate(((inp_corr_model[(*preidx1, *preidx2)], out_corr_model[(*preidx1, *preidx2)]),
                                                        (inp_noise_model[(*preidx1, *preidx2)], out_noise_model[(*preidx1, *preidx2)]))):
                        red_mask = mask_model[(*preidx1, *preidx2)]
                        l = np.arange(raw_y.size)
                        fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row',
                                                figsize=(12, 8), dpi=100, height_ratios=(2, 1),
                                                layout='constrained')
                        axs[0, 0].semilogx(l, raw_y, alpha=0.3, label='raw')
                        axs[0, 0].semilogx(l, fit_y, alpha=1, label='fit')
                        axs[0, 0].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')
                        axs[0, 0].set_ylabel(['$r_{\ell}$', '$N_{\ell}$'][i], fontsize=12)
                        axs[0, 0].legend()
                        axs[0, 0].grid()

                        axs[1, 0].semilogx(l, (fit_y - raw_y), alpha=1, color='k')
                        axs[1, 0].scatter(l[red_mask], (fit_y - raw_y)[red_mask], alpha=1, c='r')
                        axs[1, 0].set_xlabel('$\ell$', fontsize=12)
                        axs[1, 0].set_ylabel('fit - raw', fontsize=12)
                        axs[1, 0].grid()

                        axs[0, 1].plot(l, raw_y, alpha=0.3, label='raw')
                        axs[0, 1].plot(l, fit_y, alpha=1, label='fit')
                        axs[0, 1].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')

                        axs[0, 1].legend()
                        axs[0, 1].grid()

                        axs[1, 1].plot(l, (fit_y - raw_y), alpha=1, color='k')
                        axs[1, 1].scatter(l[red_mask], (fit_y - raw_y)[red_mask], alpha=1, c='r')
                        axs[1, 1].set_xlabel('$\ell$', fontsize=12)
                        axs[1, 1].grid()

                        chan1, chan2 = sv2arrs2chans[sv1][ar1][preidx1[0]], sv2arrs2chans[sv1][ar1][preidx2[0]]
                        pol1, pol2 = 'TEB'[preidx1[1]], 'TEB'[preidx2[1]]
                        fig.suptitle(f'{sv1}_{ar1}_set{split1} {chan1}_{pol1}x{chan2}_{pol2}')

                        plt_fn = f"{plot_dir}/{sv1}_{ar1}_set{split1}_{chan1}_{pol1}x{chan2}_{pol2}_{['corr', 'ps'][i]}.png"
                        plt.savefig(plt_fn, bbox_inches='tight')
                        plt.close()

        alms = None # help with memory