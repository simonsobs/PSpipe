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

def noise_matrix(noise_dir, exp, freqs, lmax, nsplits):
    
    """This function uses the noise power spectra computed by 'planck_noise_model'
    and generate a three dimensional array of noise power spectra [nfreqs,nfreqs,lmax] for temperature
    and polarisation.
    The different entries ([i,j,:]) of the arrays contain the noise power spectra
    for the different frequency channel pairs.
    for example nl_array_t[0,0,:] =>  nl^{TT}_{f_{0},f_{0}),  nl_array_t[0,1,:] =>  nl^{TT}_{f_{0},f_{1})
    this allows to have correlated noise between different frequency channels.
        
    Parameters
    ----------
    noise_data_dir : string
      the folder containing the noise power spectra
    exp : string
      the experiment to consider ('Planck')
    freqs: 1d array of string
      the frequencies we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
      nl_per_split= nl * n_{splits}
    """    
    nfreqs = len(freqs)
    nl_array_t = np.zeros((nfreqs, nfreqs, lmax))
    nl_array_pol = np.zeros((nfreqs, nfreqs, lmax))

    for count, freq in enumerate(freqs):

        l, nl_t = np.loadtxt("%s/noise_TT_mean_%s_%sx%s_%s.dat"%(noise_dir, exp, freq, exp, freq), unpack=True)
        l, nl_pol = np.loadtxt("%s/noise_EE_mean_%s_%sx%s_%s.dat"%(noise_dir, exp, freq, exp, freq), unpack=True)

        nl_array_t[count, count, :] = nl_t[:] * nsplits
        nl_array_pol[count, count, :] = nl_pol[:] * nsplits
    
    return l, nl_array_t, nl_array_pol

def generate_noise_alms(nl_array_t, nl_array_pol, lmax, n_splits):
    
    """This function generates the alms corresponding to the noise power spectra matrices
    nl_array_t, nl_array_pol. The function returns a dictionnary nlms["T", i].
    The entry of the dictionnary are for example nlms["T", i] where i is the index of the split.
    note that nlms["T", i] is a (nfreqs, size(alm)) array, it is the harmonic transform of
    the noise realisation for the different frequencies.
        
    Parameters
    ----------
    nl_array_t : 3d array [nfreq, nfreq, lmax]
      noise power spectra matrix for temperature data
    nl_array_pol : 3d array [nfreq, nfreq, lmax]
      noise power spectra matrix for polarisation data

    lmax : integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate

    """
    
    nlms = {}
    for k in range(n_splits):
        nlms["T", k] = curvedsky.rand_alm(nl_array_t, lmax=lmax)
        nlms["E", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax)
        nlms["B", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax)
    
    return nlms


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())




