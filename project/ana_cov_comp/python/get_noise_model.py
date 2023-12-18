"""
This script computes the noise model for the covariance from the measured alms,
using auto - crosses. For INKA covariances, we only need the pseudospectra.
"""
import sys

from pspipe_utils import log
from pspy import so_dict, pspy_utils

from pixell import curvedsky

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from itertools import combinations, combinations_with_replacement as cwr, product
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

alms_dir = d['alms_dir']
savgol_w = d['noise_model_savgol_w']
savgol_k = d['noise_model_savgol_k']

noise_model_dir = d["noise_model_dir"]
plot_dir = os.path.join(d["plot_dir"], "noise_model")
pspy_utils.create_directory(noise_model_dir)
pspy_utils.create_directory(plot_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

# we will make noise models for each survey and array,
# so "everything" happens inside this loop
for sv1 in surveys:
    for ar1 in arrays[sv1]:
        print(f'Doing {sv1}, {ar1}')

        # load alms for this survey and array
        # we are assuming:
        # - noise is uncorrelated between surveys 
        # - noise is uncorrelated between arrays within surveys
        # - alms have shape (npol=3, nalm)
        alms = []
        nchan = len(arrays[sv1][ar1])
        
        for i, chan1 in enumerate(arrays[sv1][ar1]):
            if i == 0:
                nsplit = len(d[f'maps_{sv1}_{ar1}_{chan1}'])
            else:
                _nsplit = len(d[f'maps_{sv1}_{ar1}_{chan1}'])
                assert _nsplit == nsplit, \
                    f'sv={sv1}, ar={ar1}, chan={chan1}, nsplit={_nsplit}, expected {nsplit}'
            
            for split1 in range(nsplit):
                alms.append(np.load(f'{alms_dir}/alms_{sv1}_{ar1}_{chan1}_{split1}.npy'))
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

        signal_model = 0
        count = 0
        for split1, split2 in combinations(range(nsplit), r=2):
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
            print(f'Doing {sv1}, {ar1}, set{split1}')

            alms1 = alms[:, split1]
            total_model = np.zeros((nchan, 3, nchan, 3, nell), dtype=alms.real.dtype)
            
            # signal model is guaranteed to be symmetric cause of split auto
            for preidx1, preidx2 in cwr(np.ndindex((nchan, 3)), r=2):
                total_model[(*preidx1, *preidx2)] = curvedsky.alm2cl(alms1[preidx1], alms1[preidx2])
                if preidx1 != preidx2:
                    total_model[(*preidx2, *preidx1)] = total_model[(*preidx1, *preidx2)]

            inp_noise_model = total_model - signal_model
            out_noise_model = np.zeros_like(inp_noise_model)

            inp_corr_model = np.zeros_like(inp_noise_model)
            out_corr_model = np.zeros_like(inp_noise_model)
            
            mask_model = np.zeros_like(inp_noise_model, dtype=bool)

            # huzzah! now we have a noisy thing that we want to smooth, 
            # nonparametrically. we need to fit the diagonals first, because
            # we will fit the off-diagonal corrs, and need to convert them back
            # to cross-spectra using the fit diagonals; i.e., the fit diagonals
            # all need to exist before doing the off-diagonals.
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
                    print(f'{preidx1=} has {np.sum(mask)} values <= 0')
                np.clip(y, 0, None, out=y)
                fit_y = savgol_filter(y, savgol_w, savgol_k)
                y[mask] = fit_y[mask]
                assert not np.any(y <= 0), \
                    print(f'{preidx1=} still has {np.sum(y <= 0)} values <= 0 after correcting')
                mask_model[(*preidx1, *preidx1)][start:] = mask

                # now fit in log space
                fit_y = savgol_filter(np.log(y), savgol_w, savgol_k)
                fit_y = np.exp(fit_y)
                out_noise_model[(*preidx1, *preidx1)][start:] = fit_y

            for preidx1, preidx2 in combinations(np.ndindex((nchan, 3)), r=2):

                # only if we are TT, we start at l=0, otherwise l=2
                if preidx1[1] == 0 and preidx2[1] == 0:
                    start = 0
                else:
                    assert np.all(inp_noise_model[(*preidx1, *preidx2)][:2] == 0), \
                        f'{preidx1=}, {preidx2=}; expected 0 for l=0,1'
                    start = 2

                y = inp_noise_model[(*preidx1, *preidx2)][start:].copy()
                y /= np.sqrt(out_noise_model[(*preidx1, *preidx1)][start:])
                y /= np.sqrt(out_noise_model[(*preidx2, *preidx2)][start:])

                inp_corr_model[(*preidx1, *preidx2)][start:] = y
                inp_corr_model[(*preidx2, *preidx1)][start:] = y

                # might start >= |1|, so first we clip and fit without arctanh, and
                # add fitted pts back where it is >= |1|. in principle
                # this could still fail, so if it doesn't, then this was a 
                # fluctuation consistent with noise
                mask = np.logical_or(y <= -1, 1 <= y) 
                if np.any(mask):
                    print(f'{preidx1=}, {preidx2=} has {np.sum(mask)} values >= |1|')
                np.clip(y, -1, 1, out=y)
                fit_y = savgol_filter(y, savgol_w, savgol_k)
                y[mask] = fit_y[mask]
                assert not np.any(np.logical_or(y <= -1, 1 <= y)), \
                    print(f'{preidx1=}, {preidx2=} still has {np.logical_or(y <= -1, 1 <= y)} values >= |1| after correcting')
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

                    chan1, chan2 = arrays[sv1][ar1][preidx1[0]], arrays[sv1][ar1][preidx2[0]]
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

                        chan1, chan2 = arrays[sv1][ar1][preidx1[0]], arrays[sv1][ar1][preidx2[0]]
                        pol1, pol2 = 'TEB'[preidx1[1]], 'TEB'[preidx2[1]]
                        fig.suptitle(f'{sv1}_{ar1}_set{split1} {chan1}_{pol1}x{chan2}_{pol2}')

                        plt_fn = f"{plot_dir}/{sv1}_{ar1}_set{split1}_{chan1}_{pol1}x{chan2}_{pol2}_{['corr', 'ps'][i]}.png"
                        plt.savefig(plt_fn, bbox_inches='tight')
                        plt.close()

            noise_model_fn = f'{noise_model_dir}/{sv1}_{ar1}_set{split1}.npy'
            np.save(noise_model_fn, out_noise_model)

        alms = None # help with memory