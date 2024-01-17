"""
This script:
1. Gets best-fit filter at the unbinned 2pt level (pol and T).
2. Gets best-fit filter at the binned, disconnected 4pt levels (pol and T).

This script assumes:
1. No cross-survey spectra.
2. All power spectra and masks are similar enough for all fields in a survey.
"""
from pspipe_utils import log
from pspy import so_dict, so_map, so_mcm, pspy_utils

from pixell import curvedsky

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import os
import sys
from functools import partial

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

filters_dir = d['filters_dir']
plot_dir = os.path.join(d['plot_dir'], 'filters')
pspy_utils.create_directory(filters_dir)
pspy_utils.create_directory(plot_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

apply_kspace_filter = d["apply_kspace_filter"]
lmax = d['lmax']
ainfo = curvedsky.alm_info(lmax=lmax)

num_tf_sims = d['num_tf_sims']

bin_low, bin_high, bin_cent, _ = pspy_utils.read_binning_file(d['binning_file'], lmax)

def get_ps(lmax, lknee, lcap, pow):
    """Get a mock power spectrum that is white at high-ell, a power-law at low
    ell, but is capped at a given minimum ell.

    Parameters
    ----------
    lmax : int
        lmax of power spectrum.
    lknee : int
        lknee of power law.
    lcap : int
        minimum ell at which the power law is capped.
    pow : scalar
        exponent of power law.

    Returns
    -------
    np.ndarray (lmax+1,)
        mock power spectrum. 
    """
    ells = np.arange(lmax + 1, dtype=np.float64)
    ps = np.zeros_like(ells)
    ps[lcap:] = (ells[lcap:]/lknee)**pow + 1
    ps[:lcap] = ps[lcap]
    return ps

def bin_spec(specs, bin_low, bin_high):
    """Bin spectra along their last axis.

    Parameters
    ----------
    specs : (..., nell) np.ndarray
        Spectra to be binned, with ell along last axis.
    bin_low : (nbin) np.ndarray
        Inclusive low-bounds of bins.
    bin_high : (nbin)
        Inclusive high-bounds of bins.

    Returns
    -------
    (..., nbin) np.ndarray
        Binned spectra.
    """
    out = np.zeros((*specs.shape[:-1], len(bin_low)))
    for i in range(len(bin_low)):
        out[..., i] = specs[..., bin_low[i]:bin_high[i] + 1].mean(axis=-1) 
    return out

def bin_mat(mats, bin_low, bin_high):
    """Bin a matrix along its last two axes.

    Parameters
    ----------
    mats : (..., nell, nell) np.ndarray
        Matrices to be binned, with ells along last two axes.
    bin_low : (nbin) np.ndarray
        Inclusive low-bounds of bins.
    bin_high : (nbin)
        Inclusive high-bounds of bins.

    Returns
    -------
    (..., nbin, nbin) np.ndarray
        Binned matrices.
    """
    out = np.zeros((*mats.shape[:-2], len(bin_low), len(bin_low)))
    for i in range(len(bin_low)):
        for j in range(len(bin_low)):
            out[..., i, j] = mats[..., bin_low[i]:bin_high[i] + 1, bin_low[j]:bin_high[j] + 1].mean(axis=(-2, -1))
    return out

def get_expected_pseudo_func(mcm, tf, ps, bin_low=None, bin_high=None):
    """Build a function that returns the theory pseudospectrum from a theory
    powerspectrum, multiplied by some one-dimensional transfer function raised
    to the alpha power:

    f(alpha): mcm @ (tf**alpha * ps)

    Parameters
    ----------
    mcm : (nell, nell) np.ndarray
        Mode-coupling matrix.
    tf : (nell)
        One-dimensional transfer function.
    ps : (nell)
        Power spectrum.
    bin_low : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin lowerbounds, by default None.
        Binning occurs if not None.
    bin_high : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin upperbounds, by default None.

    Returns
    -------
    function
        f(alpha): mcm @ (tf**alpha * ps)
    """
    if bin_low is not None:
        def f(alpha):
            return bin_spec(mcm @ (tf**alpha * ps), bin_low, bin_high)       
    else:
        def f(alpha):
            return mcm @ (tf**alpha * ps)
    return f

def get_expected_cov_diag_func(mcm, w2, tf, ps, coup, bin_low=None, bin_high=None, pre_mcm_inv=None):
    """Build a function that returns the theory covariance diagonal (under the 
    arithmetic INKA approximation) from a theory powerspectrum, multiplied by
    some one-dimensional transfer function raised to the alpha power, and the
    other covariance ingredients (mcm, w2, coup):

    f(alpha): 0.5 * ((mcm @ (tf**(alpha/2) * ps / w2)) + (mcm @ (tf**(alpha/2) * ps / w2))[:, None])**2 * coup

    Parameters
    ----------
    mcm : (nell, nell) np.ndarray
        Mode-coupling matrix.
    w2 : scalar
        w2 factor of the mask generating the mode-coupling matrix.
    tf : (nell)
        One-dimensional transfer function.
    ps : (nell)
        Power spectrum.
    coup : (nell, nell) np.ndarray
        Coupling matrix.
    bin_low : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin lowerbounds, by default None.
        Binning occurs if not None.
    bin_high : (nbin) np.ndarray, optional
        One-dimensional array of inclusive bin upperbounds, by default None.
    pseudo : bool, optional
        Whether to return the pseudospectrum covariance or powerspectrum
        covariance, by default True (pseudospectrum).
    pre_mcm_inv : (nell, nell) np.ndarray, optional
        Linear operator that takes pseudospectra to powerspectra, used in 
        calculating the powerspectrum covariance matrix, by default None.
        Returns the powerspectrum covariance matrix if not None.

    Returns
    -------
    function
        f(alpha): f(alpha): 0.5 * ((mcm @ (tf**(alpha/2) * ps / w2)) + (mcm @ (tf**(alpha/2) * ps / w2))[:, None])**2 * coup
    """
    def pseudo_cov(alpha):
        return 0.5 * ((mcm @ (tf**(alpha/2) * ps / w2)) + (mcm @ (tf**(alpha/2) * ps / w2))[:, None])**2 * coup

    if bin_low is not None:
        if pre_mcm_inv is None:
            def f(alpha):
                return np.diag(bin_mat(pseudo_cov(alpha), bin_low, bin_high))
        else:
            def f(alpha):
                return np.diag(bin_mat(pre_mcm_inv @ pseudo_cov(alpha) @ pre_mcm_inv.T, bin_low, bin_high))
    else:
        if pre_mcm_inv is None:
            def f(alpha):
                return np.diag(pseudo_cov(alpha))
        else:
            def f(alpha):
                return np.diag(pre_mcm_inv @ pseudo_cov(alpha) @ pre_mcm_inv.T)  
    
    return f

def fit_func(x, alpha, func, xmin, den):
    """A wrapper around a one-dimensional function to fit (a function of alpha
    only) that allows its result to be normalized by some denominator and only
    fit at and above some element xmin.

    Parameters
    ----------
    x : any
        x-values (not used)
    alpha : scalar
        See get_expected_pseudo_func and get_expected_cov_diag_func.
    func : function
        get_expected_pseudo_func or get_expected_cov_diag_func.
    xmin : int
        Minimum element used in the fit.
    den : np.ndarray
        Denominator used in the fitting, same size as func.

    Returns
    -------
    np.ndarray
        The normalized function result.

    Notes
    -----
    Passed to scipy.optimize.curvefit by freezing func, xmin, and den with
    functools.partial. Note, scipy.optimize.curvefit requires x-values to be
    passable, but we don't use them.
    """
    return np.divide(func(alpha)[xmin:], den[xmin:], where=den[xmin:]!=0, out=np.zeros_like(den[xmin:]))

if apply_kspace_filter:

    for sv1 in surveys:
        # get the filter tf templates lmins for the fitting
        log.info(f'Getting filter tf templates for {sv1=}')
        fl2 = np.load(f'{filters_dir}/{sv1}_fl2_fullsky.npy')
        fl4 = np.load(f'{filters_dir}/{sv1}_fl4_fullsky.npy')

        lmin2_fit = np.max(np.where(fl2 <= 0.5)[0]) + 1
        bmin2_fit = np.min(np.where(bin_low >= lmin2_fit)[0])
        lmin4_fit = np.max(np.where(fl4 <= 0.5)[0]) + 1
        bmin4_fit = np.min(np.where(bin_low >= lmin4_fit)[0])

        # Get our mock power spectra
        log.info(f'Getting mock power spectra for {sv1=}')
        
        ps_params = d[f'mock_ps_{sv1}']
        pss = {
            k: get_ps(lmax, lknee=ps_params[k]['lknee'], lcap=ps_params[k]['lcap'], pow=ps_params[k]['pow']) for k in ('T', 'pol')
        }

        # Get our average mask and couplings (total normalization doesn't matter)
        log.info(f'Getting mock mask and couplings for {sv1=}')
        
        masks = {k: so_map.read_map(f'{filters_dir}/{sv1}_win_{k}.fits') for k in ('T', 'pol')}
        for pol, mask in masks.items():
            w2 = np.sum(mask.data**2 * mask.data.pixsizemap()) / 4 / np.pi

            fn = f'{filters_dir}/{sv1}_win_{pol}_00_mcm.npy'
            if os.path.exists(fn):
                mcm = np.load(fn)
            else:
                # note we are going to calculate window alm to 2lmax, but that's ok because
                # the lmax_limit is half the Nyquist limit
                assert lmax <= mask.get_lmax_limit(), \
                    "the requested lmax is too high with respect to the map pixellisation"

                mask_alm = curvedsky.map2alm(mask.data, lmax=2*lmax, method='cyl')
                mcm = so_mcm.coupling_block('00', mask_alm.astype(np.complex128, copy=False), lmax, input_alm=True)
                mcm *= (2*np.arange(lmax + 1) + 1) / (4*np.pi)
                np.save(fn, mcm)

            fn = f'{filters_dir}/{sv1}_win_{pol}_00_mcm_inv.npy'
            if os.path.exists(fn):
                mcm_inv = np.load(fn)
            else:
                mcm_inv = np.linalg.inv(mcm)
                np.save(fn, mcm_inv)

            fn = f'{filters_dir}/{sv1}_win_{pol}_00_coupling.npy'
            if os.path.exists(fn):
                coupling = np.load(fn)
            else:
                # note we are going to calculate window alm to 2lmax, but that's ok because
                # the lmax_limit is half the Nyquist limit
                assert lmax <= mask.get_lmax_limit(), \
                    "the requested lmax is too high with respect to the map pixellisation"

                mask_alm = curvedsky.map2alm(mask.data**2, lmax=2*lmax, method='cyl')
                coupling = so_mcm.coupling_block('00', mask_alm.astype(np.complex128, copy=False), lmax, input_alm=True)
                coupling /= 4*np.pi
                np.save(fn, coupling)

            ps = pss[pol]
            
            # now we do the fits
            log.info(f'Getting 2pt and 4pt fits for {sv1=}')

            fn = f'{filters_dir}/{sv1}_{pol}_res_dict.npy'
            if os.path.exists(fn):
                res_dict = np.load(fn, allow_pickle=True).item()
            else:  
                res_dict = {}
                res_dict['lmin2_fit'] = lmin2_fit
                res_dict['bmin2_fit'] = bmin2_fit
                res_dict['lmin4_fit'] = lmin4_fit
                res_dict['bmin4_fit'] = bmin4_fit

                # get spectra
                pseudos = []
                for i in range(num_tf_sims):
                    pseudos.append(np.load(f'{filters_dir}/{sv1}_{pol}_filt_masked_spec{i}.npy'))
                pseudos = np.array(pseudos)
                binned_pseudos = bin_spec(pseudos, bin_low, bin_high)  

                pseudo_cov_diags = num_tf_sims/(num_tf_sims-1) * (pseudos - pseudos.mean(axis=0))**2
                binned_pseudo_cov_diags = num_tf_sims/(num_tf_sims-1) * (binned_pseudos - binned_pseudos.mean(axis=0))**2 

                # pseudo spectra
                func = get_expected_pseudo_func(mcm, fl2, ps)
                den = func(0)

                res_dict['pseudo_mean'] = pseudos.mean(axis=0) 
                res_dict['pseudo_var'] = pseudos.var(axis=0, ddof=1) / num_tf_sims
                res_dict['pseudo_den'] = den

                ydata = np.divide(res_dict['pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
                yerr = np.divide(res_dict['pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
                popt_pseudo, pcov_pseudo = curve_fit(partial(fit_func, func=func, xmin=lmin2_fit, den=den), 1, ydata[lmin2_fit:], sigma=yerr[lmin2_fit:])
                pseudo_best_fit = fit_func(1, popt_pseudo[0], func, 0, den)

                res_dict['pseudo_alpha'] = popt_pseudo[0]
                res_dict['pseudo_alpha_err'] = pcov_pseudo[0, 0]**0.5 
                res_dict['pseudo_best_fit'] = pseudo_best_fit
                res_dict['pseudo_err'] = ydata - pseudo_best_fit
                res_dict['pseudo_stderr'] = np.divide(ydata - pseudo_best_fit, yerr, where=yerr!=0, out=np.zeros_like(yerr))

                # binned pseudo spectra
                func = get_expected_pseudo_func(mcm, fl2, ps, bin_low=bin_low, bin_high=bin_high)
                den = func(0)

                res_dict['binned_pseudo_mean'] = binned_pseudos.mean(axis=0) 
                res_dict['binned_pseudo_var'] = binned_pseudos.var(axis=0, ddof=1) / num_tf_sims
                res_dict['binned_pseudo_den'] = den

                ydata = np.divide(res_dict['binned_pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
                yerr = np.divide(res_dict['binned_pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
                popt_binned_pseudo, pcov_binned_pseudo = curve_fit(partial(fit_func, func=func, xmin=bmin2_fit, den=den), 1, ydata[bmin2_fit:], sigma=yerr[bmin2_fit:])
                binned_pseudo_best_fit = fit_func(1, popt_binned_pseudo[0], func, 0, den)

                res_dict['binned_pseudo_alpha'] = popt_binned_pseudo[0]
                res_dict['binned_pseudo_alpha_err'] = pcov_binned_pseudo[0, 0]**0.5 
                res_dict['binned_pseudo_best_fit'] = binned_pseudo_best_fit
                res_dict['binned_pseudo_err'] = ydata - binned_pseudo_best_fit
                res_dict['binned_pseudo_stderr'] = np.divide(ydata - binned_pseudo_best_fit, yerr, where=yerr!=0, out=np.zeros_like(yerr))  

                # pseudo cov
                func4 = get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling)
                den = func4(0)

                res_dict['pseudo_cov_diag_mean'] = pseudo_cov_diags.mean(axis=0)
                res_dict['pseudo_cov_diag_var'] = pseudo_cov_diags.var(axis=0, ddof=1) / num_tf_sims
                res_dict['pseudo_cov_diag_den'] = den

                ydata = np.divide(res_dict['pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
                yerr = np.divide(res_dict['pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
                popt4, pcov4 = curve_fit(partial(fit_func, func=func4, xmin=lmin4_fit, den=den), 1, ydata[lmin4_fit:], sigma=yerr[lmin4_fit:])
                pseudo_cov_diag_best_fit4 = fit_func(1, popt4[0], func4, 0, den)

                res_dict['pseudo_cov_diag_alpha4'] = popt4[0]
                res_dict['pseudo_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
                res_dict['pseudo_cov_diag_best_fit4'] = pseudo_cov_diag_best_fit4
                res_dict['pseudo_cov_diag_err4'] = ydata - pseudo_cov_diag_best_fit4
                res_dict['pseudo_cov_diag_stderr4'] = np.divide(ydata - pseudo_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

                # binned pseudo cov
                func4 = get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling, bin_low=bin_low, bin_high=bin_high)
                den = func4(0)

                res_dict['binned_pseudo_cov_diag_mean'] = binned_pseudo_cov_diags.mean(axis=0)
                res_dict['binned_pseudo_cov_diag_var'] = binned_pseudo_cov_diags.var(axis=0, ddof=1) / num_tf_sims
                res_dict['binned_pseudo_cov_diag_den'] = den

                ydata = np.divide(res_dict['binned_pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
                yerr = np.divide(res_dict['binned_pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
                popt4, pcov4 = curve_fit(partial(fit_func, func=func4, xmin=bmin4_fit, den=den), 1, ydata[bmin4_fit:], sigma=yerr[bmin4_fit:])
                binned_pseudo_cov_diag_best_fit4 = fit_func(1, popt4[0], func4, 0, den)

                res_dict['binned_pseudo_cov_diag_alpha4'] = popt4[0]
                res_dict['binned_pseudo_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
                res_dict['binned_pseudo_cov_diag_best_fit4'] = binned_pseudo_cov_diag_best_fit4
                res_dict['binned_pseudo_cov_diag_err4'] = ydata - binned_pseudo_cov_diag_best_fit4
                res_dict['binned_pseudo_cov_diag_stderr4'] = np.divide(ydata - binned_pseudo_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

                # spectra and binned spectra
                eff_tf2 = fl2**popt_pseudo[0]
                eff_tf2_inv = np.divide(1, eff_tf2, where=eff_tf2!=0, out=np.full(eff_tf2.shape, 1_000_000, dtype=np.float64)) # 1/0 = ?
                pre_mcm_inv = np.einsum('r, rc -> rc', eff_tf2_inv, mcm_inv) # tf_inv @ mcm_inv

                specs = np.einsum('...Ll, ...il -> ...iL', pre_mcm_inv, pseudos)
                binned_specs = bin_spec(specs, bin_low, bin_high)   

                spec_cov_diags = num_tf_sims/(num_tf_sims-1) * (specs - specs.mean(axis=0))**2
                binned_spec_cov_diags = num_tf_sims/(num_tf_sims-1) * (binned_specs - binned_specs.mean(axis=0))**2
                
                # spec cov
                func4 = get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling, pre_mcm_inv=pre_mcm_inv)
                den = func4(0)
                
                res_dict['spec_cov_diag_mean'] = spec_cov_diags.mean(axis=0)
                res_dict['spec_cov_diag_var'] = spec_cov_diags.var(axis=0, ddof=1) / num_tf_sims
                res_dict['spec_cov_diag_den'] = den

                ydata = np.divide(res_dict['spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
                yerr = np.divide(res_dict['spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
                popt4, pcov4 = curve_fit(partial(fit_func, func=func4, xmin=lmin4_fit, den=den), 1, ydata[lmin4_fit:], sigma=yerr[lmin4_fit:])
                spec_cov_diag_best_fit4 = fit_func(1, popt4[0], func4, 0, den)

                res_dict['spec_cov_diag_alpha4'] = popt4[0]
                res_dict['spec_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
                res_dict['spec_cov_diag_best_fit4'] = spec_cov_diag_best_fit4
                res_dict['spec_cov_diag_err4'] = ydata - spec_cov_diag_best_fit4
                res_dict['spec_cov_diag_stderr4'] = np.divide(ydata - spec_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

                # binned spec cov
                func4 = get_expected_cov_diag_func(mcm, w2, fl4, ps, coupling, pre_mcm_inv=pre_mcm_inv, bin_low=bin_low, bin_high=bin_high)
                den = func4(0)

                res_dict['binned_spec_cov_diag_mean'] = binned_spec_cov_diags.mean(axis=0)
                res_dict['binned_spec_cov_diag_var'] = binned_spec_cov_diags.var(axis=0, ddof=1) / num_tf_sims
                res_dict['binned_spec_cov_diag_den'] = den

                ydata = np.divide(res_dict['binned_spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
                yerr = np.divide(res_dict['binned_spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
                popt4, pcov4 = curve_fit(partial(fit_func, func=func4, xmin=bmin4_fit, den=den), 1, ydata[bmin4_fit:], sigma=yerr[bmin4_fit:])
                binned_spec_cov_diag_best_fit4 = fit_func(1, popt4[0], func4, 0, den)

                res_dict['binned_spec_cov_diag_alpha4'] = popt4[0]
                res_dict['binned_spec_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
                res_dict['binned_spec_cov_diag_best_fit4'] = binned_spec_cov_diag_best_fit4
                res_dict['binned_spec_cov_diag_err4'] = ydata - binned_spec_cov_diag_best_fit4
                res_dict['binned_spec_cov_diag_stderr4'] = np.divide(ydata - binned_spec_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

                np.save(f'{filters_dir}/{sv1}_{pol}_res_dict.npy', res_dict)

            # fit pseudo
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 6), height_ratios=(2, 1), sharey='row', sharex='col')

            den = res_dict['pseudo_den']
            ydata = np.divide(res_dict['pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
            yerr = np.divide(res_dict['pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))

            pseudo_best_fit = res_dict['pseudo_best_fit']
            pseudo_err = res_dict['pseudo_err']
            pseudo_stderr = res_dict['pseudo_stderr']

            pseudo_alpha = res_dict['pseudo_alpha']
            pseudo_alpha_err = res_dict['pseudo_alpha_err']

            axs[0, 0].plot(ydata)
            axs[0, 0].plot(pseudo_best_fit, label=rf'$\alpha_s={pseudo_alpha:.3f} \pm {pseudo_alpha_err:.3f}$')
            axs[0, 0].axvspan(0, lmin2_fit, edgecolor='none', facecolor='k', alpha=0.2)
            axs[0, 0].legend(loc='lower right')
            axs[0, 0].grid()
            axs[0, 0].set_ylabel(r'$\tilde{\mathcal{C}_{\ell}}(\alpha) / \tilde{\mathcal{C}_{\ell}}(0)$')
            axs[0, 0].set_title('unbinned')

            axs[1, 0].plot(pseudo_err / pseudo_best_fit, label=f'$\chi^2={np.mean(pseudo_stderr[lmin2_fit:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((pseudo_err[lmin2_fit:] / pseudo_best_fit[lmin2_fit:])**2)**0.5:.3f}$')
            axs[1, 0].axvspan(0, lmin2_fit, edgecolor='none', facecolor='k', alpha=0.2)
            axs[1, 0].set_ylim(-.05, .05)
            axs[1, 0].legend(loc='lower right')
            axs[1, 0].grid()
            axs[1, 0].set_xlabel('$\ell$')
            axs[1, 0].set_ylabel('$\Delta y_s / y_s$')

            den = res_dict['binned_pseudo_den']
            ydata = np.divide(res_dict['binned_pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
            yerr = np.divide(res_dict['binned_pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
            
            binned_pseudo_best_fit = res_dict['binned_pseudo_best_fit']
            binned_pseudo_err = res_dict['binned_pseudo_err']
            binned_pseudo_stderr = res_dict['binned_pseudo_stderr']

            binned_pseudo_alpha = res_dict['binned_pseudo_alpha']
            binned_pseudo_alpha_err = res_dict['binned_pseudo_alpha_err']

            axs[0, 1].plot(bin_cent, ydata)
            axs[0, 1].plot(bin_cent, binned_pseudo_best_fit, label=rf'$\alpha_s={binned_pseudo_alpha:.3f} \pm {binned_pseudo_alpha_err:.3f}$')
            axs[0, 1].axvspan(0, bin_cent[bmin2_fit], edgecolor='none', facecolor='k', alpha=0.2)
            axs[0, 1].legend(loc='lower right')
            axs[0, 1].grid()
            axs[0, 1].set_title('binned')

            axs[1, 1].plot(bin_cent, binned_pseudo_err / binned_pseudo_best_fit, label=f'$\chi^2={np.mean(binned_pseudo_stderr[bmin2_fit:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((binned_pseudo_err[bmin2_fit:] / binned_pseudo_best_fit[bmin2_fit:])**2)**0.5:.3f}$')
            axs[1, 1].axvspan(0, bin_cent[bmin2_fit], edgecolor='none', facecolor='k', alpha=0.2)
            axs[1, 1].set_ylim(-.05, .05)
            axs[1, 1].legend(loc='lower right')
            axs[1, 1].grid()
            axs[1, 1].set_xlabel('$\ell$')

            fig.suptitle('pseudo specs')
            fig.tight_layout()
            fig.savefig(f'{plot_dir}/{sv1}_{pol}_pseudo_fit.png')

            # fit pseudo_cov_diag
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 6), height_ratios=(2, 1), sharey='row', sharex='col')

            den = res_dict['pseudo_cov_diag_den']
            ydata = np.divide(res_dict['pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
            yerr = np.divide(res_dict['pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))

            pseudo_cov_diag_best_fit4 = res_dict['pseudo_cov_diag_best_fit4']
            pseudo_cov_diag_err4 = res_dict['pseudo_cov_diag_err4']
            pseudo_cov_diag_stderr4 = res_dict['pseudo_cov_diag_stderr4']

            pseudo_cov_diag_alpha4 = res_dict['pseudo_cov_diag_alpha4']
            pseudo_cov_diag_alpha4_err = res_dict['pseudo_cov_diag_alpha4_err']

            axs[0, 0].plot(ydata)
            axs[0, 0].plot(pseudo_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={pseudo_cov_diag_alpha4/pseudo_alpha:.3f} \pm {pseudo_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
            axs[0, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
            axs[0, 0].legend(loc='lower right')
            axs[0, 0].grid()
            axs[0, 0].set_ylabel(r'$\tilde{\Sigma_{\ell,\ell}}(\alpha) / \tilde{\Sigma_{\ell,\ell}}(0)$')
            axs[0, 0].set_title('unbinned')

            axs[1, 0].plot(pseudo_cov_diag_err4 / pseudo_cov_diag_best_fit4, label=f'$\chi_4^2={np.mean(pseudo_cov_diag_stderr4[lmin4_fit:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((pseudo_cov_diag_err4[lmin4_fit:] / pseudo_cov_diag_best_fit4[lmin4_fit:])**2)**0.5:.3f}$')
            axs[1, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
            axs[1, 0].set_ylim(-.1, .1)
            axs[1, 0].legend(loc='lower right')
            axs[1, 0].grid()
            axs[1, 0].set_xlabel('$\ell$')
            axs[1, 0].set_ylabel('$\Delta y_4 / y_4$')

            den = res_dict['binned_pseudo_cov_diag_den']
            ydata = np.divide(res_dict['binned_pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
            yerr = np.divide(res_dict['binned_pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
            
            binned_pseudo_cov_diag_best_fit4 = res_dict['binned_pseudo_cov_diag_best_fit4']
            binned_pseudo_cov_diag_err4 = res_dict['binned_pseudo_cov_diag_err4']
            binned_pseudo_cov_diag_stderr4 = res_dict['binned_pseudo_cov_diag_stderr4']

            binned_pseudo_cov_diag_alpha4 = res_dict['binned_pseudo_cov_diag_alpha4']
            binned_pseudo_cov_diag_alpha4_err = res_dict['binned_pseudo_cov_diag_alpha4_err']

            axs[0, 1].errorbar(bin_cent, ydata, yerr, linestyle='none')
            axs[0, 1].plot(bin_cent, binned_pseudo_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={binned_pseudo_cov_diag_alpha4/pseudo_alpha:.3f} \pm {binned_pseudo_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
            axs[0, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
            axs[0, 1].legend(loc='lower right')
            axs[0, 1].grid()
            axs[0, 1].set_title('binned')

            axs[1, 1].errorbar(bin_cent, binned_pseudo_cov_diag_err4 / binned_pseudo_cov_diag_best_fit4, yerr / binned_pseudo_cov_diag_best_fit4, linestyle='none', label=f'$\chi_4^2={np.mean(binned_pseudo_cov_diag_stderr4[bmin4_fit:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((binned_pseudo_cov_diag_err4[bmin4_fit:] / binned_pseudo_cov_diag_best_fit4[bmin4_fit:])**2)**0.5:.3f}$')
            axs[1, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
            axs[1, 1].set_ylim(-.1, .1)
            axs[1, 1].legend(loc='lower right')
            axs[1, 1].grid()
            axs[1, 1].set_xlabel('$\ell$')

            fig.suptitle('pseudo cov diags')
            fig.tight_layout()
            fig.savefig(f'{plot_dir}/{sv1}_{pol}_pseudo_cov_diag_fit.png')

            # fit spec_cov_diag
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 6), height_ratios=(2, 1), sharey='row', sharex='col')

            den = res_dict['spec_cov_diag_den']
            ydata = np.divide(res_dict['spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
            yerr = np.divide(res_dict['spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
            
            spec_cov_diag_best_fit4 = res_dict['spec_cov_diag_best_fit4']
            spec_cov_diag_err4 = res_dict['spec_cov_diag_err4']
            spec_cov_diag_stderr4 = res_dict['spec_cov_diag_stderr4']

            spec_cov_diag_alpha4 = res_dict['spec_cov_diag_alpha4']
            spec_cov_diag_alpha4_err = res_dict['spec_cov_diag_alpha4_err']

            axs[0, 0].plot(ydata)
            axs[0, 0].plot(spec_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={spec_cov_diag_alpha4/pseudo_alpha:.3f} \pm {spec_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
            axs[0, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
            axs[0, 0].legend(loc='lower right')
            axs[0, 0].grid()
            axs[0, 0].set_ylabel(r'$\tilde{\Sigma_{\ell,\ell}}(\alpha) / \tilde{\Sigma_{\ell,\ell}}(0)$')
            axs[0, 0].set_title('unbinned')

            axs[1, 0].plot(spec_cov_diag_err4 / spec_cov_diag_best_fit4, label=f'$\chi_4^2={np.mean(spec_cov_diag_stderr4[lmin4_fit:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((spec_cov_diag_err4[lmin4_fit:] / spec_cov_diag_best_fit4[lmin4_fit:])**2)**0.5:.3f}$')
            axs[1, 0].axvspan(0, lmin4_fit, edgecolor='none', facecolor='k', alpha=0.2)
            axs[1, 0].set_ylim(-.1, .1)
            axs[1, 0].legend(loc='lower right')
            axs[1, 0].grid()
            axs[1, 0].set_xlabel('$\ell$')
            axs[1, 0].set_ylabel('$\Delta y_4 / y_4$')

            den = res_dict['binned_spec_cov_diag_den']
            ydata = np.divide(res_dict['binned_spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
            yerr = np.divide(res_dict['binned_spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))

            binned_spec_cov_diag_best_fit4 = res_dict['binned_spec_cov_diag_best_fit4']
            binned_spec_cov_diag_err4 = res_dict['binned_spec_cov_diag_err4']
            binned_spec_cov_diag_stderr4 = res_dict['binned_spec_cov_diag_stderr4']

            binned_spec_cov_diag_alpha4 = res_dict['binned_spec_cov_diag_alpha4']
            binned_spec_cov_diag_alpha4_err = res_dict['binned_spec_cov_diag_alpha4_err']

            axs[0, 1].errorbar(bin_cent, ydata, yerr, linestyle='none')
            axs[0, 1].plot(bin_cent, binned_spec_cov_diag_best_fit4, label=rf'$\alpha_4/\alpha_s={binned_spec_cov_diag_alpha4/pseudo_alpha:.3f} \pm {binned_spec_cov_diag_alpha4_err/pseudo_alpha:.3f}$', color='C1')
            axs[0, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
            axs[0, 1].legend(loc='lower right')
            axs[0, 1].grid()
            axs[0, 1].set_title('binned')

            axs[1, 1].errorbar(bin_cent, binned_spec_cov_diag_err4 / binned_spec_cov_diag_best_fit4, yerr / binned_spec_cov_diag_best_fit4, linestyle='none', label=f'$\chi_4^2={np.mean(binned_spec_cov_diag_stderr4[bmin4_fit:]**2):.3f}$, $\%\mathrm{{rms}}={100 * np.mean((binned_spec_cov_diag_err4[bmin4_fit:] / binned_spec_cov_diag_best_fit4[bmin4_fit:])**2)**0.5:.3f}$')
            axs[1, 1].axvspan(0, bin_cent[bmin4_fit], edgecolor='none', facecolor='k', alpha=0.2)
            axs[1, 1].set_ylim(-.1, .1)
            axs[1, 1].legend(loc='lower right')
            axs[1, 1].grid()
            axs[1, 1].set_xlabel('$\ell$')

            fig.suptitle('cov diags')
            fig.tight_layout()
            fig.savefig(f'{plot_dir}/{sv1}_{pol}_spec_cov_diag_fit.png')
else:
    log.info(f'WARNING: no kspace filter, so this {__name__} is unnecessary')