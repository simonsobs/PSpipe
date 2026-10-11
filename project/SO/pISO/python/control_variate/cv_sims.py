"""Control-variate simulations: the "real" and "alt" branches.

Both start from the same white pixel noise n_p on a CAR patch (T, Q, U independent):

    real = map2alm[ W alm2map( sqrt(C_l) n_lm ) ]      colour, then mask
    alt  = sqrt(C_l) map2alm( W n_p )                  mask, then colour

with n_lm = A map2alm(n_p). The noise is drawn on the patch only, so n_lm has power f_sky
rather than 1, and A = 1 / sqrt(f_sky) restores it. alt has an analytic covariance
(cv_covariance.py); real does not.
"""
import healpy as hp
import numpy as np
from pspy import pspy_utils, so_spectra, sph_tools

SPECTRA = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


def sqrt_cl_matrix(theory_file, lmax):
    """Matrix square root of the (T, E, B) spectra, with TE, as a (3, 3, lmax + 1) array."""
    _, ps = pspy_utils.ps_lensed_theory_to_dict(theory_file, "Cl", lmax=lmax, start_at_zero=True)
    n = min(lmax + 1, len(ps["TT"]))
    cl = np.zeros((3, 3, lmax + 1))
    cl[0, 0, :n], cl[1, 1, :n], cl[2, 2, :n] = ps["TT"][:n], ps["EE"][:n], ps["BB"][:n]
    cl[0, 1, :n] = cl[1, 0, :n] = ps["TE"][:n]
    cl[:, :, :2] = 0
    sqrt_cl = np.zeros_like(cl)
    for l in range(2, lmax + 1):
        w, v = np.linalg.eigh(cl[:, :, l])
        sqrt_cl[:, :, l] = v @ np.diag(np.sqrt(np.maximum(w, 0))) @ v.T
    return sqrt_cl


def make_branches(template, win_T, win_pol, sqrt_cl, lmax, niter=0):
    """Return draw(rng), real(n_p), alt(n_p) and A for a 3-component CAR `template` map."""
    pixsize = template.data.pixsizemap()
    A = 1 / np.sqrt(pixsize.sum() / (4 * np.pi))

    def draw(rng):
        n_p = template.copy()
        n_p.data[...] = rng.standard_normal(n_p.data.shape) / np.sqrt(pixsize)
        return n_p

    def mask(m):
        out = m.copy()
        out.data[0] = m.data[0] * win_T.data
        out.data[1] = m.data[1] * win_pol.data
        out.data[2] = m.data[2] * win_pol.data
        return out

    def colour(alm):
        out = np.zeros_like(alm)
        for i in range(3):
            for j in range(3):
                out[i] += hp.almxfl(alm[j].astype(np.complex128), sqrt_cl[i, j])
        return out

    def alt(n_p):
        return colour(A * sph_tools.map2alm(mask(n_p), niter, lmax))

    def real(n_p):
        n_lm = A * sph_tools.map2alm(n_p, niter, lmax)
        sky = sph_tools.alm2map(colour(n_lm), template.copy())
        return sph_tools.map2alm(mask(sky), niter, lmax)

    return draw, real, alt, A


def bandpowers(alm, lmax, binning_file, spectra=SPECTRA):
    """Binned pseudo-spectra of a (T, E, B) alm array, as {spectrum: array}."""
    ell = np.arange(lmax + 1)
    _, ps = so_spectra.get_spectra_pixell(alm, alm, spectra=SPECTRA)
    return {s: so_spectra.bin_spectra(ell, ps[s], binning_file, lmax, "Cl")[1] for s in spectra}
