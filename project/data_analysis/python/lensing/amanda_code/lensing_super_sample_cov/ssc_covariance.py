import os
from copy import deepcopy
import numpy as np
from pixell import enmap, curvedsky as cs
import healpy


def get_sigma_kappa_squared(mask, lmax, clkk):
    """
    Returns sigma_kappa^2, the rms fluctuation of the lensing convergence
    field within the survey area, from eq. 3 in Manzotti et. al.
    (arXiv:1401.7992).

    Parameters
    ----------
    mask : pixell.enmap.ndmap
        The spatial window function, including any point source masks
    lmax : int
        The maximum lensing multipole to use in the calculation.
    clkk : array of float
        The theory lensing convergence spectrum,
        C_L^kk = [L(L+1)]^2 C_L^phiphi / 4.

    Returns
    -------
    sigmakappa2 : float
        The variance of the lensing convergence field within the patch.
    """
    w_lm = cs.map2alm(mask, lmax=lmax, tweak=True)
    # we need to sum over all L's and M's of |w_LM|^2 * C_L^kk:
    w_lm_clkk = healpy.almxfl(w_lm, np.sqrt(clkk[:lmax+1]))
    w_lm_clkk_sq = np.real(np.conjugate(w_lm_clkk) * w_lm_clkk) # |w_LM * sqrt(C_L^kk)|^2
    # take sum of |w_LM * sqrt(C_L^kk)|^2 over all L's and all M >= 0;
    # multiply by 2 to include all M <= 0, which double-counts M=0,
    # so then subtract one of the M=0 terms:
    sigmakappa2_sum = 2 * np.real(np.sum(w_lm_clkk_sq))
    sidx = healpy.Alm.getidx(lmax, 0, 0)
    eidx = healpy.Alm.getidx(lmax, lmax, 0)
    sigmakappa2_sum -= np.real(np.sum(w_lm_clkk_sq[sidx:eidx+1]))
    # now divide by the area^2:
    area = mask.sum()/mask.size * mask.area() # are of non-zero pixels, in steradians
    sigmakappa2 = sigmakappa2_sum / area**2
    return sigmakappa2


def calculate_cmb_derivs(lensed_theo, ell_ranges, cmb_spectra=['tt', 'te', 'ee', 'bb']):
    """
    Returns a dictionary of derivatives of the lenssed CMB TT, TE, EE, BB
    power spectra, (1 / ell^2) * d[ell^2 C_ell] / d[ln(ell)].

    Parameters
    ----------
    lensed_theo : dict of array of float
        A dictionary holding the ** LENSED ** CMB theory spectra, in units
        of uK^2 as C_l's (no ell-factors), with keys `'tt'`, `'te'`, `'ee'`,
        and `'bb'`.  All spectra are expected to start at ell = 0.
    ell_ranges : dict of list of int
        A dictionary with keys `'tt'`, `'te'`, `'ee'`, `'bb'` holding a
        list `[lmin, lmax]` of the multipole limits for each spectrum.
        For example, `ell_ranges = {'tt': [500, 8000], 'ee': [300, 8000]}`.
    cmb_spectra : list of str, default=['tt', 'te', 'ee', 'bb']
        A list of CMB spectra to calculate derivatives for. The names 
        use the same format as the keys of `lensed_theo`.

    Returns
    -------
    derivs : dict of array of float
        A dictionary holding the derivatives, with the keys given by 
        the elements of `cmb_spectra`.
    """
    # calculate (1 / ell^2) * d[ell^2 C_ell] / d[ln(ell)]
    # or equivalently, (1 / ell) * d[ell^2 C_ell] / d[ell]
    derivs = {}
    for s in ['tt', 'te', 'ee', 'bb']:
        lmax = ell_ranges[s][1]
        ells = np.arange(lmax+1)
        deriv = np.roll(ells**2 * lensed_theo[s][:lmax+1], 1) - ells**2 * lensed_theo[s][:lmax+1]
        derivs[s] = np.zeros(lmax+1)
        derivs[s][2:] = deriv[2:] / ells[2:]
    return derivs


def calculate_ssc_blocks(mask, lensed_theo, ell_ranges, Lmax, cmb_spectra=['tt', 'te', 'ee', 'bb']):
    """
    Returns a dictionary holding the SSC term for the blocks of the
    covariance matrix.

    Parameters
    ----------
    mask : pixell.enmap.ndmap
        The spatial window function, including any point source masks.
    lensed_theo : dict of array of float
        A dictionary holding the ** LENSED ** CMB theory spectra and
        CMB lensing potential (phi) or convergence (kappa) spectrum.
        The CMB spectra should be in units of uK^2 as C_l's (no ell-factors),
        and have keys `'tt'`, `'te'`, `'ee'`, `'bb'`. The CMB lensing
        potential spectrum C_L^phiphi should have a key `'pp'`, and/or you
        can pass the CMB lensing convergence spectrum with a key `'kk'`
        using the convention C_L^kk = [L(L+1)]^2 C_L^phiphi / 4. All spectra
        are expected to start at ell = 0 and end at the same multipole.
    ell_ranges : dict of list of int
        A dictionary with keys `'tt'`, `'te'`, `'ee'`, `'bb'` holding a
        list `[lmin, lmax]` of the multipole limits for each spectrum.
        For example, `ell_ranges = {'tt': [500, 8000], 'ee': [300, 8000]}`.
    Lmax : int
        The maximum multipole to use when calculating the variance of the
        lensing convergence field within the survey area, sigma_kappa^2.
    cmb_spectra : list of str, default=['tt', 'te', 'ee', 'bb']
        A list of CMB spectra to calculate derivatives for. The names 
        use the same format as the keys of `lensed_theo`.

    Returns
    -------
    ssc_blocks : dict of dict of array of float
        A nested dictionary holding the SSC term for the blocks of the
        covariance matrix. Both sets of keys are the names of the CMB
        spectra: `'tt'`, `'te'`, `'ee'`, and `'bb'`. Each block begins
        at ell = 0, with zeros filled in below the minimum multipole
        for each spectrum. For example, `ssc_blocks['tt']['ee']` has a
        shape (lmaxTT+1, lmaxEE+1).
    """
    if 'kk' in lensed_theo.keys():
        clkk = lensed_theo['kk'][:Lmax+1].copy()
    elif 'pp' in lensed_theo.keys():
        L = np.arange(Lmax + 1)
        clkk = (L * (L + 1))**2 * lensed_theo['pp'][:Lmax+1] / 4
    sigmakappa2 = get_sigma_kappa_squared(mask, Lmax, clkk)
    derivs = calculate_cmb_derivs(lensed_theo, ell_ranges, cmb_spectra=cmb_spectra)
    ssc_blocks = {s: {} for s in cmb_spectra}
    for i, s1 in enumerate(cmb_spectra):
        lmin1 = ell_ranges[s1][0]
        lmax1 = ell_ranges[s1][1]
        for s2 in cmb_spectra[i:]:
            lmin2 = ell_ranges[s2][0]
            lmax2 = ell_ranges[s2][1]
            ssc_blocks[s1][s2] = np.zeros((lmax1+1, lmax2+1))
            ssc_cov = np.outer(derivs[s1][lmin1:lmax1+1], derivs[s2][lmin2:lmax2+1]) * sigmakappa2
            ssc_blocks[s1][s2][lmin1:, lmin2:] = ssc_cov.copy()
            if s2 != s1:
                ssc_blocks[s2][s1] = np.transpose(ssc_blocks[s1][s2])
    return ssc_blocks


