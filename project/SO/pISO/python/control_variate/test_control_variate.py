"""Tests of the analytic covariance of the control-variate branch.

    python test_control_variate.py        (or pytest; about 15 s)

- array bookkeeping: with cross-array correlation rho, Cov(ab, ab) / Cov(aa, aa) = (1 + rho^2) / 2
- two identical arrays reproduce the one-array covariance
- the covariance with the spin-weighted kernels equals the exact one, obtained by writing the
  control as a linear map of the pixel noise (no Monte Carlo): C = xi^T B xi gives
  Cov(C, C') = 2 <B, B'>
"""
import os
import sys

import healpy as hp
import numpy as np
from pspy import so_cov, so_map, so_window, sph_tools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_covariance import SqrtPS, sigma_ll
from cv_sims import sqrt_cl_matrix
from cv_spin2_kernels import spin2_kernels

THEORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "../../../../old/Planck_pspy/data/cosmo2017_10K_acc3_lensedCls.dat")


def _patch(res=90.0):
    """A rectangular CAR patch and a function giving apodized windows on it."""
    t1 = so_map.car_template(1, -38, 38, -19, 19, res)
    t3 = so_map.car_template(3, -38, 38, -19, 19, res)

    def window(pad, apod):
        b = t1.copy()
        b.data[:] = 0
        b.data[pad:-pad, pad:-pad] = 1
        return so_window.create_apodization(b, "C1", apod, use_rmax=True)

    return t1, t3, window


def _matrix_sqrt(m):
    """Matrix square root over the first two axes of a symmetric (n, n, nl) array, for l >= 2."""
    out = np.zeros_like(m)
    for l in range(2, m.shape[-1]):
        w, v = np.linalg.eigh(m[:, :, l])
        out[:, :, l] = v @ np.diag(np.sqrt(np.maximum(w, 0))) @ v.T
    return out


def _flat_sqrt(rho, lmax, narrays=2):
    """sqrt(C) for flat unit spectra with cross-array correlation rho and no field mixing."""
    m = np.zeros((3 * narrays, 3 * narrays, lmax + 1))
    for p in range(3):
        for i in range(narrays):
            for j in range(narrays):
                m[i + narrays * p, j + narrays * p, 2:] = 1.0 if i == j else rho
    return _matrix_sqrt(m)


def test_array_correlation():
    lmax = 80
    _, _, window = _patch()
    W = window(3, 4.5)
    coup = so_cov.cov_coupling_spin0and2_simple(
        {f"{k}{leg}": W for leg in "abcd" for k in "TP"}, lmax, niter=0)
    for rho in (0.0, 0.3, 0.6, 0.9, 1.0):
        sp = SqrtPS(_flat_sqrt(rho, lmax), ["a", "b"], lmin=2, lmax=lmax)

        def var(ij, pq):
            return np.diag(sigma_ll(sp, coup, "TT", "TT", ij, pq))[30]

        aaaa = var(("a", "a"), ("a", "a"))
        assert abs(var(("a", "b"), ("a", "b")) / aaaa - (1 + rho ** 2) / 2) < 1e-10
        assert abs(var(("a", "a"), ("b", "b")) / aaaa - rho ** 2) < 1e-10


def test_identical_arrays_collapse():
    lmax = 60
    _, _, window = _patch()
    W = window(3, 4.5)
    coup = so_cov.cov_coupling_spin0and2_simple(
        {f"{k}{leg}": W for leg in "abcd" for k in "TP"}, lmax, niter=0)
    sq = sqrt_cl_matrix(THEORY, lmax)
    cl = np.einsum("ijl,jkl->ikl", sq, sq)
    big = np.zeros((6, 6, lmax + 1))                           # two arrays with identical spectra
    for p in range(3):
        for q in range(3):
            for a in range(2):
                for b in range(2):
                    big[a + 2 * p, b + 2 * q] = cl[p, q]
    one = SqrtPS.from_single_array(sq, "a", lmin=2, lmax=lmax)
    two = SqrtPS(_matrix_sqrt(big), ["a", "b"], lmin=2, lmax=lmax)
    for spec in ("TT", "EE", "TE"):
        ref = sigma_ll(one, coup, spec, spec)
        auto = sigma_ll(two, coup, spec, spec, ("a", "a"), ("a", "a"))
        cross = sigma_ll(two, coup, spec, spec, ("a", "b"), ("a", "b"))
        assert np.abs(auto - ref).max() / np.abs(ref).max() < 1e-10
        assert np.abs(cross - ref).max() / np.abs(ref).max() < 1e-10


def test_matches_exact_covariance():
    lmax, dl = 58, 20
    t1, t3, window = _patch()
    pix = t1.data.pixsizemap()
    A = 1 / np.sqrt(pix.sum() / (4 * np.pi))
    WT, WP = window(3, 4.5), window(5, 6.0)                      # different T and polarization windows
    ny, nx = WT.data.shape
    npix = ny * nx
    sq = sqrt_cl_matrix(THEORY, lmax)

    # the control is q_g = G_g xi, with xi = (xi_T, xi_Q, xi_U) the unit pixel noise
    rows = [(l, m) for l in range(2, lmax + 1) for m in range(l + 1)]
    ell = np.array([r[0] for r in rows])
    mm = np.array([r[1] for r in rows])
    idx = np.array([hp.Alm.getidx(lmax, l, m) for l, m in rows])
    scale = {0: (A * WT.data / np.sqrt(pix)).ravel(), 1: (A * WP.data / np.sqrt(pix)).ravel()}
    G = {g: np.zeros((len(rows), 3 * npix), complex) for g in "TEB"}
    for p in range(npix):
        for comp in range(3):
            m = t3.copy()
            m.data[:] = 0
            m.data[comp, p // nx, p % nx] = 1.0
            alm = sph_tools.map2alm(m, 0, lmax)
            col, sc = comp * npix + p, scale[min(comp, 1)][p]
            if comp == 0:
                G["T"][:, col] = alm[0][idx] * sc
            else:
                G["E"][:, col], G["B"][:, col] = alm[1][idx] * sc, alm[2][idx] * sc
    H = {X: sum(sq["TEB".index(X), j][ell][:, None] * G["TEB"[j]] for j in range(3)) for X in "TEB"}

    bins = [(lo, min(lo + dl - 1, lmax - 1)) for lo in range(2, lmax - 1, dl)]

    def quad_form(spec, lo, hi):
        X, Y = spec
        sel = (ell >= lo) & (ell <= hi)
        d = np.where(mm == 0, 1.0, 2.0)[sel] / ((2 * ell[sel] + 1) * (hi - lo + 1))
        M = ((H[Y][sel].conj().T * d) @ H[X][sel]).real
        return 0.5 * (M + M.T)

    sqrt_ps = SqrtPS.from_single_array(sq, "a", lmin=2, lmax=lmax)
    K = spin2_kernels(WT, WP, lmax)
    lc = np.arange(2, lmax)
    pairs = [("TT", "TT"), ("EE", "EE"), ("BB", "BB"), ("TE", "TE"), ("EB", "EB"), ("EE", "BB")]
    for s1, s2 in pairs:
        formula = sigma_ll(sqrt_ps, K, s1, s2, A_norm=A)
        for b1, b2 in ((bins[0], bins[0]), (bins[1], bins[1]), (bins[0], bins[1])):
            exact = 2 * np.sum(quad_form(s1, *b1) * quad_form(s2, *b2))
            sel1, sel2 = (lc >= b1[0]) & (lc <= b1[1]), (lc >= b2[0]) & (lc <= b2[1])
            got = formula[np.ix_(sel1, sel2)].mean()
            assert abs(got / exact - 1) < 0.03, (s1, s2, b1, b2, got / exact)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
