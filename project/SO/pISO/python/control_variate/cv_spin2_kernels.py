"""Coupling kernels of the control-variate covariance, with spin-weighted 3j symbols.

pspy's cov_coupling_spin0and2_simple (planck=False) uses the spin-0 kernel for every spin
structure, which is too high for polarization at low l. The exact kernels are sums over L of
the window cross-power X_L times Wigner 3j symbols, with p = (-1)^(l + l' + L):

    00  sum_L X_L (l l' L; 0 0 0)^2
    0+  sum_L X_L (l l' L; 0 0 0)(l l' L; 2 -2 0)            (+0 is the same)
    ++  sum_L X_L (l l' L; 2 -2 0)^2 (1 + p) / 2
    --  sum_L X_L (l l' L; 2 -2 0)^2 (1 - p) / 2

X_L = (2L + 1) / (4 pi) times the cross-power of alm(W_a W_c) and alm(W_b W_d). cov_coupling_
spin0and2_simple(..., planck=True) gives 00, 0+, +0 and ++; only the odd-parity `--` kernel
is new here. One window per spin type: T legs use win_T and polarization legs win_pol.
"""
import healpy as hp
import numpy as np
from ducc0.misc import wigner3j_int
from pspy import sph_tools

from cv_covariance import KERNEL_FOR


def _cross_power(win_a, win_b, lmax, niter):
    def alm_of_square(w):
        sq = w.copy()
        sq.data[...] = np.asarray(w.data) ** 2
        return sph_tools.map2alm(sq, niter, lmax)

    cl = hp.alm2cl(alm_of_square(win_a), alm_of_square(win_b))[:lmax + 1]
    ell = np.arange(lmax + 1)
    return cl * (2 * ell + 1) / (4 * np.pi)


def spin2_kernels(win_T, win_pol, lmax, niter=0):
    """Kernels keyed like cv_covariance.KERNEL_FOR, as (lmax - 2, lmax - 2) arrays (l = 2..lmax-1)."""
    x00 = _cross_power(win_T, win_T, lmax, niter)
    x0p = _cross_power(win_T, win_pol, lmax, niter)
    xpp = _cross_power(win_pol, win_pol, lmax, niter)
    n = lmax - 2
    K = {k: np.zeros((n, n)) for k in ("00", "0+", "++", "--")}
    for i in range(n):
        for j in range(i, n):
            l, lp = i + 2, j + 2
            L0, g0 = wigner3j_int(l, lp, 0, 0)
            _, g2 = wigner3j_int(l, lp, 2, -2)
            L = L0 + np.arange(len(g0))
            keep = L <= lmax
            L, g0, g2 = L[keep], g0[keep], g2[keep]
            even = ((l + lp + L) % 2 == 0).astype(float)
            K["00"][i, j] = np.sum(x00[L] * g0 ** 2)
            K["0+"][i, j] = np.sum(x0p[L] * g0 * g2)
            K["++"][i, j] = np.sum(xpp[L] * g2 ** 2 * even)
            K["--"][i, j] = np.sum(xpp[L] * g2 ** 2 * (1 - even))
    for k in K:
        K[k] = K[k] + K[k].T - np.diag(np.diag(K[k]))
    K["+0"] = K["0+"].copy()
    return {KERNEL_FOR[s]: v for s, v in K.items()}
