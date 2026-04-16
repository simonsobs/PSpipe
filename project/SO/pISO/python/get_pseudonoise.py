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
from pspipe_utils import log, pspipe_list
from pspy import so_dict, so_mpi, so_spectra, pspy_utils

from pixell import curvedsky

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from itertools import combinations, combinations_with_replacement as cwr, product
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

alms_dir = d['alms_dir']
lmax = d["lmax"]
savgol_w = d['savgol_w']
savgol_k = d['savgol_k']

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
square2flat_spectra = ['TT', 'TE', 'TB', 'ET', 'EE', 'EB', 'BT', 'BE', 'BB']

bestfit_dir = d["best_fits_dir"]
noise_dir = opj(bestfit_dir, 'noise')
plot_dir = opj(d['plots_dir'], 'noise')
pspy_utils.create_directory(noise_dir)
pspy_utils.create_directory(plot_dir)

# we want all the possible cross-spectra, even between tubes where we expect the
# noise correlation to be small (however, we skip survey crosses). later, when
# adding blocks, we can opt to not include tube-tube correlations
_, sv1_list, m1_list, sv2_list, m2_list = pspipe_list.get_spectra_list(d)
sv1_list = np.array(sv1_list)
m1_list = np.array(m1_list)
sv2_list = np.array(sv2_list)
m2_list = np.array(m2_list)

equal_sv_idxs = np.where(sv1_list == sv2_list)[0]
n_cross = len(equal_sv_idxs)
sv1_list = sv1_list[equal_sv_idxs]
m1_list = m1_list[equal_sv_idxs]
sv2_list = sv2_list[equal_sv_idxs]
m2_list = m2_list[equal_sv_idxs]

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_cross - 1)
log.info(f"[Rank {so_mpi.rank}] number of cross-map pairs to compute: {len(subtasks)}")

for task in subtasks:
    sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]
    assert sv1 == sv2, f'How did we get {sv1=} != {sv2=}?'
    sv = sv1
    spec_name = f"{sv}_{m1}x{sv}_{m2}"

    log.info(f"[{task:02d}] Computing raw noise pseudospectra for {spec_name}")
    
    # load alms for this survey and map pair
    # we are assuming:
    # - noise is uncorrelated between surveys 
    # - alms have shape (npol=3, nalm)
    alms1 = []
    alms2 = []

    nsplits = d[f"n_splits_{sv}"]
    for k in range(nsplits):
        alms1.append(np.load(f"{alms_dir}" + f"alms_{sv}_{m1}_set{k}.npy"))
        alms2.append(np.load(f"{alms_dir}" + f"alms_{sv}_{m2}_set{k}.npy"))
    
    alms1 = np.asarray(alms1)
    alms2 = np.asarray(alms2)

    assert alms1.ndim == 3, f'{alms1.ndim=}, expected 3'
    assert alms1.shape[:2] == (nsplits, 3), f'{alms1.shape[:2]=}, expected ({nsplits}, 3)'
    assert alms1.shape == alms2.shape, f'{alms1.shape=} != {alms2.shape=}'

    # get the signal model for this survey and map pair from
    # the average of split cross spectra. 
    # we are assuming:
    # - noise is uncorrelated between splits
    splits_cross_iterator = pspipe_list.get_splits_cross_iterator(sv, nsplits, sv, nsplits)

    nell = curvedsky.nalm2lmax(alms1.shape[-1]) + 1
    assert nell == lmax + 1, f'Expected nell={lmax + 1}, got {nell=}'

    # NOTE: redundant computation is performed when sv1==sv2 and m1==m2, but the
    # code is cleaner
    signal_model = np.zeros((3, 3, nell), dtype=np.float32)
    for k1, k2 in splits_cross_iterator: # 01 02 03, 10 12 13, ... 
        for p1, p2 in product(range(3), repeat=2):
            signal_model[p1, p2] += curvedsky.alm2cl(alms1[k1, p1], alms2[k2, p2])
    signal_model /= len(splits_cross_iterator)
    
    # for each pair of maps we measure the spectrum for each split separately
    # we are assuming:
    # - noise may be correlated between any maps
    # - beams, masks, cals and poleffs, etc (anything modifying the signal) are
    # the same for each split
    for k in range(nsplits):
        total_model = np.zeros((3, 3, nell), dtype=np.float32)
        for p1, p2 in product(range(3), repeat=2):
            total_model[p1, p2] = curvedsky.alm2cl(alms1[k, p1], alms2[k, p2])
        inp_noise_model = total_model - signal_model

        inp_noise_model = inp_noise_model[..., 2:-1] # TODO: reconsider pspy convention
        
        # save as human-readable spectrum, so l convention and Cl type is clear
        l = np.arange(2, lmax, dtype=inp_noise_model.dtype)
        so_spectra.write_ps_matrix(opj(noise_dir, f'raw_pseudo_noise_{spec_name}_set{k}.dat'),
                                   l, inp_noise_model, 'Cl', spectra=spectra)

    alms1 = None
    alms2 = None

# huzzah! now we have a noisy thing that we want to smooth, 
# nonparametrically. we want to fit the diagonals first, because we will fit the
# off-diagonal corrs, and want to use our estimate of the "true" autos to
# convert cross spectra from spectra to/from correlations. 
#
# because of mpi, we need to wait for each raw pseudo to be calculated first so 
# that crosses know they can safely load the autos.
#
# NOTE: this also means the smoothed autos will be recalculated many times (for
# every cross that loads it), but this stuff is so fast that it's fine
so_mpi.barrier()

# if m1 == m2, we have something like this, which is on the main diagonal:
# | TT_mm TE_mm TB_mm |
# | ET_mm EE_mm EB_mm | 
# | BT_mm BE_mm BB_mm | 
# so the diagonals of inp_noise_model are actually diagonals, while the other
# elements are corrs. when iterating, do diags first, then upper corrs, then
# plot diags + upper corrs

# if m1 != m2, we have something like this, where everything is a corr:
# | TT_m1m2 TE_m1m2 TB_m1m2 |
# | ET_m1m2 EE_m1m2 EB_m1m2 | 
# | BT_m1m2 BE_m1m2 BB_m1m2 | 
# so we need to separately load the corresponding diagonals for each element.
# when iterating, do the corresponding diags first, then all elements, then
# plot all elements

for task in subtasks:
    sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]
    assert sv1 == sv2, f'How did we get {sv1=} != {sv2=}?'
    sv = sv1
    spec_name = f"{sv}_{m1}x{sv}_{m2}" # cross, if m1!=m2, auto if m1==m2

    log.info(f"[{task:02d}] Computing smoothed noise pseudospectra for {spec_name}")

    if m1 != m2:
        spec_name1 = f"{sv}_{m1}x{sv}_{m1}" # auto for m1
        spec_name2 = f"{sv}_{m2}x{sv}_{m2}" # auto for m2
    
    nsplits = d[f"n_splits_{sv}"]
    for k in range(nsplits):
        _, inp_noise_model = so_spectra.read_ps_matrix(opj(noise_dir, f'raw_pseudo_noise_{spec_name}_set{k}.dat'),
                                                       spectra=spectra,
                                                       return_type='Cl',
                                                       return_dtype=np.float32)
        out_noise_model = np.zeros_like(inp_noise_model)

        if m1 != m2:
            _, inp_noise_model1 = so_spectra.read_ps_matrix(opj(noise_dir, f'raw_pseudo_noise_{spec_name1}_set{k}.dat'),
                                                            spectra=spectra,
                                                            return_type='Cl',
                                                            return_dtype=np.float32)
            _, inp_noise_model2 = so_spectra.read_ps_matrix(opj(noise_dir, f'raw_pseudo_noise_{spec_name2}_set{k}.dat'),
                                                            spectra=spectra,
                                                            return_type='Cl',
                                                            return_dtype=np.float32)
            diags = np.zeros((2, 3, nell - 3), dtype=inp_noise_model.dtype)
        
        # for plots
        inp_corr_model = np.zeros_like(inp_noise_model)
        out_corr_model = np.zeros_like(inp_noise_model)
        mask_model = np.zeros_like(inp_noise_model, dtype=bool)

        # autos (will be placed in diags array if m1!=m2)
        if m1 == m2:
            inms = (inp_noise_model,)
        else:
            inms = (inp_noise_model1, inp_noise_model2)
        for m_idx, inm in enumerate(inms):
            for p in range(3):
                y = inm[p, p].copy()

                # might start <= 0, so first we clip and fit without log, and
                # add fitted pts back where it is <= 0. in principle
                # this could still fail, so if it doesn't, then this was a 
                # fluctuation consistent with noise
                mask = y <= 0
                if np.any(mask):
                    log.info(f'[{task:02d}] {m_idx=}, {p=} has {np.sum(mask)} values <= 0, clipping and setting to fit values')
                np.clip(y, 0, None, out=y)
                fit_y = savgol_filter(y, savgol_w, savgol_k)
                y[mask] = fit_y[mask]
                if np.any(y <= 0):
                    log.info(f'[{task:02d}] {m_idx=}, {p=} still has {np.sum(y <= 0)} values <= 0 after correcting, setting to min of good values')
                    y[y <= 0] = np.min(y[y > 0]) # mask already includes all these points
                mask_model[p, p] = np.logical_or(mask_model[p, p], mask)

                # now fit in log space
                fit_y = savgol_filter(np.log(y), savgol_w, savgol_k)
                fit_y = np.exp(fit_y)

                if m1 == m2:
                    out_noise_model[p, p] = fit_y
                else:
                    diags[m_idx, p] = fit_y

        # crosses
        # fit the correlations, not the raw spectra because it's easier to keep
        # the values to physical values (r=-1 to 1)
        if m1 == m2:
            pol_iterator = list(combinations(range(3), r=2))
        else:
            pol_iterator = list(product(range(3), repeat=2))
        for p1, p2 in pol_iterator:
            y = inp_noise_model[p1, p2].copy()

            # normalize by the auto fits instead of the raw autos
            # because the fit is smooth / what we think is "the truth"
            if m1 == m2:
                y /= np.sqrt(out_noise_model[p1, p1])
                y /= np.sqrt(out_noise_model[p2, p2]) 
            else:
                y /= np.sqrt(diags[0, p1]) 
                y /= np.sqrt(diags[1, p2])

            # first do fit on corrs, get out_corr_model
            # then turn out_corr_model into out_noise_model
            inp_corr_model[p1, p2] = y
            if m1 == m2:
                inp_corr_model[p2, p1] = y

            # might start >= |1|, so first we clip and fit without arctanh, and
            # add fitted pts back where it is >= |1|. in principle
            # this could still fail, so if it doesn't, then this was a 
            # fluctuation consistent with noise
            mask = np.logical_or(y <= -1, 1 <= y) 
            if np.any(mask):
                log.info(f'[{task:02d}] {p1=}, {p2=} has {np.sum(mask)} values >= |1|, clipping and setting to fit values')
            np.clip(y, -1, 1, out=y)
            fit_y = savgol_filter(y, savgol_w, savgol_k)
            y[mask] = fit_y[mask]
            if np.any(np.logical_or(y <= -1, 1 <= y)):
                log.info(f'[{task:02d}] {p1=}, {p2=} still has {np.sum(np.logical_or(y <= -1, 1 <= y))} values >= |1| after correcting, setting to extremal good value')
                y[y <= -1] = np.min(y[y > -1]) # mask already includes all these points
                y[1 <= y] = np.max(y[1 > y]) # mask already includes all these points
            mask_model[p1, p2] = np.logical_or(mask_model[p1, p2], mask)
            if m1 == m2:
                mask_model[p2, p1] = mask_model[p1, p2]

            # now fit in arctanh space
            fit_y = savgol_filter(np.arctanh(y), savgol_w, savgol_k)
            fit_y = np.tanh(fit_y)
            out_corr_model[p1, p2] = fit_y
            if m1 == m2:
                out_corr_model[p2, p1] = out_corr_model[p1, p2]

            if m1 == m2:
                fit_y *= np.sqrt(out_noise_model[p1, p1])
                fit_y *= np.sqrt(out_noise_model[p2, p2])
            else:
                fit_y *= np.sqrt(diags[0, p1]) 
                fit_y *= np.sqrt(diags[1, p2]) 
            
            out_noise_model[p1, p2] = fit_y
            if m1 == m2:
                out_noise_model[p2, p1] = out_noise_model[p1, p2]

        # save as human-readable spectrum, so l convention and Cl type is clear
        l = np.arange(2, lmax, dtype=out_noise_model.dtype)
        so_spectra.write_ps_matrix(opj(noise_dir, f'pseudo_noise_{spec_name}_set{k}.dat'),
                                   l, out_noise_model, 'Cl', spectra=spectra)

        # plot and save
        if m1 == m2:
            pol_iterator = list(cwr(range(3), r=2))
        for p1, p2 in pol_iterator:
            if (m1 == m2) and (p1 == p2):
                # ps
                raw_y = inp_noise_model[p1, p2]
                fit_y = out_noise_model[p1, p2]
                l = np.arange(raw_y.size)
                red_mask = mask_model[p1, p2]
                fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row',
                                        figsize=(12, 8), dpi=100, height_ratios=(2, 1),
                                        layout='constrained')

                axs[0, 0].loglog(l, raw_y, alpha=0.3, label='raw')
                axs[0, 0].loglog(l, fit_y, alpha=1, label='fit')
                axs[0, 0].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')
                axs[0, 0].set_ylabel(r'$N_{\ell}$', fontsize=12)
                axs[0, 0].legend()
                axs[0, 0].grid()

                axs[1, 0].semilogx(l, (fit_y / raw_y), alpha=1, color='k')
                axs[1, 0].scatter(l[red_mask], (fit_y / raw_y)[red_mask], alpha=1, c='r')
                axs[1, 0].set_xlabel(r'$\ell$', fontsize=12)
                axs[1, 0].set_ylabel('fit / raw', fontsize=12)
                axs[1, 0].grid()

                axs[0, 1].semilogy(l, raw_y, alpha=0.3, label='raw')
                axs[0, 1].semilogy(l, fit_y, alpha=1, label='fit')
                axs[0, 1].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')
                axs[0, 1].legend()
                axs[0, 1].grid()

                axs[1, 1].plot(l, (fit_y / raw_y), alpha=1, color='k')
                axs[1, 1].scatter(l[red_mask], (fit_y / raw_y)[red_mask], alpha=1, c='r')
                axs[1, 1].set_xlabel(r'$\ell$', fontsize=12)
                axs[1, 1].grid()

                pol1, pol2 = 'TEB'[p1], 'TEB'[p2]
                fig.suptitle(f'{sv}_{m1}_{pol1}x{m2}_{pol2}_set{k}')

                plt_fn = f'{plot_dir}/{sv}_{m1}_{pol1}x{m2}_{pol2}_set{k}_ps.png'
                plt.savefig(plt_fn, bbox_inches='tight')
                plt.close()

            else:
                # corrs, ps
                for i, (raw_y, fit_y) in enumerate(((inp_corr_model[p1, p2], out_corr_model[p1, p2]),
                                                    (inp_noise_model[p1 ,p2], out_noise_model[p1, p2]))):
                    l = np.arange(raw_y.size)
                    red_mask = mask_model[p1, p2]
                    fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row',
                                            figsize=(12, 8), dpi=100, height_ratios=(2, 1),
                                            layout='constrained')
                    
                    axs[0, 0].semilogx(l, raw_y, alpha=0.3, label='raw')
                    axs[0, 0].semilogx(l, fit_y, alpha=1, label='fit')
                    axs[0, 0].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')
                    axs[0, 0].set_ylabel([r'$r_{\ell}$', r'$N_{\ell}$'][i], fontsize=12)
                    axs[0, 0].legend()
                    axs[0, 0].grid()

                    axs[1, 0].semilogx(l, (fit_y - raw_y), alpha=1, color='k')
                    axs[1, 0].scatter(l[red_mask], (fit_y - raw_y)[red_mask], alpha=1, c='r')
                    axs[1, 0].set_xlabel(r'$\ell$', fontsize=12)
                    axs[1, 0].set_ylabel('fit - raw', fontsize=12)
                    axs[1, 0].grid()

                    axs[0, 1].plot(l, raw_y, alpha=0.3, label='raw')
                    axs[0, 1].plot(l, fit_y, alpha=1, label='fit')
                    axs[0, 1].scatter(l[red_mask], fit_y[red_mask], alpha=1, c='r')

                    axs[0, 1].legend()
                    axs[0, 1].grid()

                    axs[1, 1].plot(l, (fit_y - raw_y), alpha=1, color='k')
                    axs[1, 1].scatter(l[red_mask], (fit_y - raw_y)[red_mask], alpha=1, c='r')
                    axs[1, 1].set_xlabel(r'$\ell$', fontsize=12)
                    axs[1, 1].grid()

                    pol1, pol2 = 'TEB'[p1], 'TEB'[p2]
                    fig.suptitle(f'{sv}_{m1}_{pol1}x{m2}_{pol2}_set{k}')

                    plt_fn = f"{plot_dir}/{sv}_{m1}_{pol1}x{m2}_{pol2}_set{k}_{['corr', 'ps'][i]}.png"
                    plt.savefig(plt_fn, bbox_inches='tight')
                    plt.close()