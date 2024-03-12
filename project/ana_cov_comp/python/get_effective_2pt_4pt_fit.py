"""
The anisotropy of the kspace filter and/or the noise breaks the standard
master toolkit, both at the 2pt (mode-coupling) and 4pt (covariance) level.
This is the third of 3 scripts that develop an ansatz correction for this.

The ansatz essentially consists of a diagonal transfer function template 
in ell raised to a power, where the power is a function of the filter, mask,
and power spectra. In this third script, we fit the mock/simple simulations
from the second script, using the templates from the first script raised to
some power (we fit for the power -- that is, we perform a single paramater
fit). 

The output of these three scripts -- the templates raised to the best-fit
powers -- are then used in two places in the rest of the pipeline. The 
two point fit is used to turn the measured noise pseudospectra into 
noise power spectra (the noise model). The four point fit is then used to
turn the signal and noise power spectra into signal and noise pseudospectra
for the covariance.

This script assumes:
1. No cross-survey spectra.
2. All power spectra and masks are similar enough for all fields in a survey.
"""
from pspipe_utils import log, pspipe_list, covariance as psc
from pspy import so_dict, so_map, so_mcm, pspy_utils

from pixell import curvedsky

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# FIXME: allow job array over channels/pols

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

filters_dir = d['filters_dir']
plot_dir = os.path.join(d['plot_dir'], 'filters')
pspy_utils.create_directory(filters_dir)
pspy_utils.create_directory(plot_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

apply_kspace_filter = d["apply_kspace_filter"]
lmax_pseudocov = d['lmax_pseudocov']
assert lmax_pseudocov >= d['lmax'], \
    f"{lmax_pseudocov=} must be >= {d['lmax']=}" 

num_tf_sims = d['num_tf_sims']

bin_low, bin_high, bin_cent, _ = pspy_utils.read_binning_file(d['binning_file'], lmax_pseudocov)

if apply_kspace_filter:

    for sv1 in sv2arrs2chans:
        # get the filter tf templates lmins for the fitting
        log.info(f'Getting filter tf templates for {sv1=}')
        fl2 = np.load(f'{filters_dir}/{sv1}_fl_2pt_fullsky.npy')
        fl4 = np.load(f'{filters_dir}/{sv1}_fl_4pt_fullsky.npy')

        lmin2_fit = np.max(np.where(fl2 <= 0.5)[0]) + 1 # 1 more than the last element of fl2 that is <= 0.5
        bmin2_fit = np.min(np.where(bin_low >= lmin2_fit)[0]) # the first bin such that bin_low >= lmin2_fit
        lmin4_fit = np.max(np.where(fl4 <= 0.5)[0]) + 1
        bmin4_fit = np.min(np.where(bin_low >= lmin4_fit)[0])

        # Get our mock power spectra
        log.info(f'Getting mock power spectra for {sv1=}')
        
        ps_params = d[f'mock_ps_{sv1}']
        pss = {
            k: psc.get_mock_noise_ps(lmax_pseudocov, lknee=ps_params[k]['lknee'], lcap=ps_params[k]['lcap'], pow=ps_params[k]['pow']) for k in ('T', 'pol')
        }

        # Get our average mask and couplings (total normalization doesn't matter)        
        masks = {k: so_map.read_map(f'{filters_dir}/{sv1}_win_{k}.fits') for k in ('T', 'pol')}
        for pol1, mask in masks.items():
            log.info(f'Getting mock mask and couplings for {sv1=}, {pol1=}')
            
            w2 = np.sum(mask.data**2 * mask.data.pixsizemap()) / 4 / np.pi

            fn = f'{filters_dir}/{sv1}_win_{pol1}_00_mcm.npy'
            if os.path.exists(fn):
                mcm = np.load(fn)
            else:
                # note we are going to calculate window alm to 2lmax, but that's ok because
                # the lmax_limit is half the Nyquist limit
                assert lmax_pseudocov <= mask.get_lmax_limit(), \
                    "the requested lmax is too high with respect to the map pixellisation"

                mask_alm = curvedsky.map2alm(mask.data, lmax=2*lmax_pseudocov, method='cyl')
                mcm = so_mcm.coupling_block('00', mask_alm.astype(np.complex128, copy=False), lmax_pseudocov, input_alm=True)
                mcm *= (2*np.arange(lmax_pseudocov + 1) + 1) / (4*np.pi)
                np.save(fn, mcm)

            fn = f'{filters_dir}/{sv1}_win_{pol1}_00_mcm_inv.npy'
            if os.path.exists(fn):
                mcm_inv = np.load(fn)
            else:
                mcm_inv = np.linalg.inv(mcm)
                np.save(fn, mcm_inv)

            fn = f'{filters_dir}/{sv1}_win_{pol1}_00_coupling.npy'
            if os.path.exists(fn):
                coupling = np.load(fn)
            else:
                # note we are going to calculate window alm to 2lmax, but that's ok because
                # the lmax_limit is half the Nyquist limit
                assert lmax_pseudocov <= mask.get_lmax_limit(), \
                    "the requested lmax is too high with respect to the map pixellisation"

                mask_alm = curvedsky.map2alm(mask.data**2, lmax=2*lmax_pseudocov, method='cyl')
                coupling = so_mcm.coupling_block('00', mask_alm.astype(np.complex128, copy=False), lmax_pseudocov, input_alm=True)
                coupling /= 4*np.pi
                np.save(fn, coupling)

            ps = pss[pol1]
            
            # now we do the fits
            log.info(f'Getting 2pt and 4pt fits for {sv1=}, {pol1=}')

            fn = f'{filters_dir}/{sv1}_{pol1}_res_dict.npy'
            if os.path.exists(fn):
                res_dict = np.load(fn, allow_pickle=True).item()
            else:  
                res_dict = {}
                res_dict['lmin2_fit'] = lmin2_fit
                res_dict['bmin2_fit'] = bmin2_fit
                res_dict['lmin4_fit'] = lmin4_fit
                res_dict['bmin4_fit'] = bmin4_fit

                # get simulated pseudo spectra and pseudo covariances
                pseudo_specs = []
                for i in range(num_tf_sims):
                    pseudo_specs.append(np.load(f'{filters_dir}/{sv1}_{pol1}_filt_masked_spec{i}.npy'))
                pseudo_specs = np.array(pseudo_specs)
                binned_pseudo_specs = psc.bin_spec(pseudo_specs, bin_low, bin_high)  

                pseudo_covs = num_tf_sims/(num_tf_sims-1) * (pseudo_specs - pseudo_specs.mean(axis=0))**2 # the mean of this is the sample variance
                binned_pseudo_covs = num_tf_sims/(num_tf_sims-1) * (binned_pseudo_specs - binned_pseudo_specs.mean(axis=0))**2 

                # pseudo spectra
                # NOTE: this is the fit that will be used downstream in constructing the noise model
                func = psc.get_expected_pseudo_func(mcm, fl2, ps)
                target = pseudo_specs
                tag = 'pseudo_spec'
                xmin = lmin2_fit

                psc.get_alpha_fit(res_dict, func, target, tag, xmin=xmin)

                # binned pseudo spectra
                func = psc.get_expected_pseudo_func(mcm, fl2, ps, bin_low=bin_low, bin_high=bin_high)
                target = binned_pseudo_specs
                tag = 'binned_pseudo_spec'
                xmin = bmin2_fit
                
                psc.get_alpha_fit(res_dict, func, target, tag, xmin=xmin)

                # pseudo cov
                func = psc.get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling)
                target = pseudo_covs
                tag = 'pseudo_cov'
                xmin = lmin2_fit                
                
                psc.get_alpha_fit(res_dict, func, target, tag, xmin=xmin)

                # binned pseudo cov
                func = psc.get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling, bin_low=bin_low, bin_high=bin_high)
                target = binned_pseudo_covs
                tag = 'binned_pseudo_cov'
                xmin = bmin2_fit                
                
                psc.get_alpha_fit(res_dict, func, target, tag, xmin=xmin)
                
                # get simulated power spectra and power covariances
                # NOTE: we need to have performed the pseudo_spec fit first in order to build the 
                # linear operator that takes pseudo spectra to power spectra
                eff_tf2 = fl2**res_dict['pseudo_spec_alpha']
                eff_tf2_inv = np.divide(1, eff_tf2, where=eff_tf2!=0, out=np.full(eff_tf2.shape, 1_000_000, dtype=np.float64)) # 1/0 = ?
                cl2dl = np.arange(lmax_pseudocov + 1) * (np.arange(lmax_pseudocov + 1) + 1) / 2 / np.pi
                pre_mcm_inv = np.einsum('r, rc -> rc', cl2dl*eff_tf2_inv, mcm_inv) # cl2dl @ tf_inv @ mcm_inv

                power_specs = np.einsum('...Ll, ...il -> ...iL', pre_mcm_inv, pseudo_specs)
                binned_power_specs = psc.bin_spec(power_specs, bin_low, bin_high)   

                power_covs = num_tf_sims/(num_tf_sims-1) * (power_specs - power_specs.mean(axis=0))**2 # the mean of this is the sample variance
                binned_power_covs = num_tf_sims/(num_tf_sims-1) * (binned_power_specs - binned_power_specs.mean(axis=0))**2
                
                # power cov
                func = psc.get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling, pre_mcm_inv=pre_mcm_inv)
                target = power_covs
                tag = 'power_cov'
                xmin = lmin2_fit                
                
                psc.get_alpha_fit(res_dict, func, target, tag, xmin=xmin)

                # binned power cov
                # NOTE: this is the fit that will be used downstream in constructing the actual covariance
                func = psc.get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling, pre_mcm_inv=pre_mcm_inv, bin_low=bin_low, bin_high=bin_high)
                target = binned_power_covs
                tag = 'binned_power_cov'
                xmin = bmin2_fit                
                
                psc.get_alpha_fit(res_dict, func, target, tag, xmin=xmin)

                np.save(f'{filters_dir}/{sv1}_{pol1}_res_dict.npy', res_dict)

            # plot the pseudo spec fits, the pseudo cov fits, and the power cov fits
            for tag in ['pseudo_spec', 'pseudo_cov', 'power_cov']:
                fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 6), height_ratios=(2, 1), sharey='row', sharex='col')

                # unbinned
                den = res_dict[f'{tag}_den']      

                ydata = res_dict[f'{tag}_ydata']
                yerr = res_dict[f'{tag}_yerr']

                xmin = res_dict[f'{tag}_xmin']
                
                alpha = res_dict[f'{tag}_alpha']
                alpha_err = res_dict[f'{tag}_alpha_err']
                best_fit = res_dict[f'{tag}_best_fit']
                best_fit_err = res_dict[f'{tag}_best_fit_err']
                best_fit_stderr = res_dict[f'{tag}_best_fit_stderr']

                # data and fit
                axs[0, 0].plot(ydata)
                axs[0, 0].plot(best_fit, label=rf'$\alpha={alpha:.3f} \pm {alpha_err:.3f}$')
                axs[0, 0].axvspan(0, xmin, edgecolor='none', facecolor='k', alpha=0.2)
                axs[0, 0].legend(loc='lower right')
                axs[0, 0].grid()
                axs[0, 0].set_ylabel(r'$y=f(\alpha) / f(0)$')
                axs[0, 0].set_title('unbinned')

                # residual
                axs[1, 0].plot(best_fit_err / best_fit, label=f'$\chi^2={np.mean(best_fit_stderr[xmin:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((best_fit_err[xmin:] / best_fit[xmin:])**2)**0.5:.3f}$')
                axs[1, 0].axvspan(0, xmin, edgecolor='none', facecolor='k', alpha=0.2)
                axs[1, 0].set_ylim(-.15, .15)
                axs[1, 0].legend(loc='lower right')
                axs[1, 0].grid()
                axs[1, 0].set_xlabel('$\ell$')
                axs[1, 0].set_ylabel('$\Delta y / y$')

                # binned
                den = res_dict[f'binned_{tag}_den']      

                ydata = res_dict[f'binned_{tag}_ydata']
                yerr = res_dict[f'binned_{tag}_yerr']

                xmin = res_dict[f'binned_{tag}_xmin']
                
                alpha = res_dict[f'binned_{tag}_alpha']
                alpha_err = res_dict[f'binned_{tag}_alpha_err']
                best_fit = res_dict[f'binned_{tag}_best_fit']
                best_fit_err = res_dict[f'binned_{tag}_best_fit_err']
                best_fit_stderr = res_dict[f'binned_{tag}_best_fit_stderr']

                # data and fit
                axs[0, 1].errorbar(bin_cent, ydata, yerr, linestyle='none')
                axs[0, 1].plot(bin_cent, best_fit, label=rf'$\alpha={alpha:.3f} \pm {alpha_err:.3f}$')
                axs[0, 1].axvspan(0, bin_cent[xmin], edgecolor='none', facecolor='k', alpha=0.2)
                axs[0, 1].legend(loc='lower right')
                axs[0, 1].grid()
                axs[0, 1].set_title('binned')

                # residual
                axs[1, 1].errorbar(bin_cent, best_fit_err / best_fit, yerr / best_fit, linestyle='none', label=f'$\chi^2={np.mean(best_fit_stderr[xmin:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((best_fit_err[xmin:] / best_fit[xmin:])**2)**0.5:.3f}$')
                axs[1, 1].axvspan(0, bin_cent[xmin], edgecolor='none', facecolor='k', alpha=0.2)
                axs[1, 1].set_ylim(-.15, .15)
                axs[1, 1].legend(loc='lower right')
                axs[1, 1].grid()
                axs[1, 1].set_xlabel('$\ell$')

                fig.suptitle(tag)
                fig.tight_layout()
                fig.savefig(f'{plot_dir}/{sv1}_{pol1}_{tag}_fit.png')
else:
    log.info(f'WARNING: no kspace filter, so this {__name__} is unnecessary')