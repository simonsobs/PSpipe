'''
Some utility functions for processing the planck data
'''

import numpy as np
import healpy as hp
from pspy import so_mcm, so_spectra, pspy_utils
from pixell import curvedsky


def process_planck_spectra(l, cl, binning_file, lmax, spectra, mcm_inv):
    
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    n_bins = len(bin_hi)
    fac = (l * (l + 1) / (2 * np.pi))
    unbin_vec = []
    mcm_inv = so_mcm.coupling_dict_to_array(mcm_inv)
    for spec in spectra:
        unbin_vec = np.append(unbin_vec, cl[spec][2:lmax])
    cl = so_spectra.vec2spec_dict(lmax-2, np.dot(mcm_inv, unbin_vec), spectra)
    l = np.arange(2, lmax)
    vec = []
    for spec in spectra:
        binnedPower = np.zeros(len(bin_c))
        for ibin in range(n_bins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
            binnedPower[ibin] = (cl[spec][loc] * fac[loc]).mean()/(fac[loc].mean())
        vec = np.append(vec, binnedPower)
    Db = so_spectra.vec2spec_dict(n_bins, vec, spectra)
    return l, cl, bin_c, Db


def subtract_mono_di(map_in, mask_in, nside):
    
    map_masked = hp.ma(map_in)
    map_masked.mask = (mask_in < 1)
    mono, dipole = hp.pixelfunc.fit_dipole(map_masked)
    print(mono, dipole)
    m = map_in.copy()
    npix = hp.nside2npix(nside)
    bunchsize = npix // 24
    bad = hp.UNSEEN
    for ibunch in range(npix // bunchsize):
        ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
        ipix = ipix[(np.isfinite(m.flat[ipix]))]
        x, y, z = hp.pix2vec(nside, ipix, False)
        m.flat[ipix] -= dipole[0] * x
        m.flat[ipix] -= dipole[1] * y
        m.flat[ipix] -= dipole[2] * z
        m.flat[ipix] -= mono
    return m

def symmetrize(a):
    
    return a + a.T - np.diag(a.diagonal())

def binning(l, cl, lmax, binning_file=None, size=None):
    
    if binning_file is not None:
        bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    else:
        bin_lo = np.arange(2, lmax, size)
        bin_hi = bin_lo + size - 1
        bin_c = (bin_lo + bin_hi) / 2
    
    fac = (l * (l + 1) / (2 * np.pi))
    n_bins = len(bin_hi)
    binnedPower = np.zeros(len(bin_c))
    for ibin in range(n_bins):
        loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
        binnedPower[ibin] = (cl[loc]*fac[loc]).mean()/(fac[loc].mean())
    return bin_c, binnedPower





