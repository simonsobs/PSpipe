"""Analytic covariance of the pseudo-spectra of the control-variate branch `alt`.

alt = sqrt(C) q with q = A map2alm(W n_p) built from white noise, so <q q*> is the mask
coupling and the covariance of two spectra is a sum over the two Wick pairings, each a sum
over five spin structures:

    Cov(C^{PiPj}_{AiAj}(l), C^{PpPq}_{ApAq}(l'))
        = A^4 sum_{Ag,Ah} sum_spin [ PI(spin; ac, bd) + PI(spin; ad, bc) ] XI^spin_{ll'}

PI is a product of sqrt(C) entries (T / E+B / E-B combinations of the two contracted pairs)
and XI the coupling kernel of that spin structure (see cv_spin2_kernels.py). Ag, Ah are the
intermediate arrays of the cross-array sqrt(C); one array gives a single term.
"""
import numpy as np

SPINS = ("00", "0+", "+0", "++", "--")

# window combination of so_cov.cov_coupling_spin0and2_simple that carries each spin structure
KERNEL_FOR = {"00": "TaTcTbTd", "0+": "TaTcPbPd", "+0": "PaPcTbTd",
              "++": "PaPcPbPd", "--": "PaPdPbPc"}

FIELDS = "TEB"


class SqrtPS:
    """Matrix square root of the cross-array, cross-field spectra.

    `sqrt_mat` is indexed [a + narrays * p, b + narrays * q, l], as returned by
    mnms.utils.np_eigpow(total_ps_mat, 0.5, axes=[0, 1]); `arrays` gives the array order.
    """

    def __init__(self, sqrt_mat, arrays, lmin=2, lmax=None):
        self.m, self.arrays, self.lmin = sqrt_mat, list(arrays), lmin
        self.n = len(self.arrays)
        self.lmax = sqrt_mat.shape[-1] if lmax is None else lmax
        if sqrt_mat.shape[:2] != (self.n * 3, self.n * 3):
            raise ValueError(f"sqrt_mat is {sqrt_mat.shape[:2]}, expected {(self.n * 3,) * 2}")

    def __call__(self, P1, P2, A1, A2):
        i = self.arrays.index(A1) + self.n * FIELDS.index(P1)
        j = self.arrays.index(A2) + self.n * FIELDS.index(P2)
        return self.m[i, j][self.lmin:self.lmax]

    @classmethod
    def from_single_array(cls, sqrt_cl, name="a", lmin=2, lmax=None):
        """Wrap a (3, 3, nl) sqrt(C) from a single array."""
        return cls(np.asarray(sqrt_cl), [name], lmin=lmin, lmax=lmax)


def ps_product(sqrt_ps, spin, spec1, spec2, pair1, pair2, Ag, Ah):
    """Spectrum factor of the `spin` kernel, as an (nl, nl) outer product over l, l'.

    spec1, spec2 are the contracted field pairs (e.g. 'TE') and pair1, pair2 the matching
    array pairs; Ag, Ah are the intermediate arrays.
    """
    (Pi, Pj), (Pp, Pq) = spec1, spec2
    (Ai, Aj), (Ap, Aq) = pair1, pair2

    def outer(Pa, Aa, Pb, Ab, mid, f1, f2):
        return sqrt_ps(Pa, f1, Aa, mid)[:, None] * sqrt_ps(Pb, f2, Ab, mid)

    def T(Pa, Aa, Pb, Ab, mid):
        return outer(Pa, Aa, Pb, Ab, mid, "T", "T")

    def plus(Pa, Aa, Pb, Ab, mid):
        return outer(Pa, Aa, Pb, Ab, mid, "E", "E") + outer(Pa, Aa, Pb, Ab, mid, "B", "B")

    def minus(Pa, Aa, Pb, Ab, mid):
        return outer(Pa, Aa, Pb, Ab, mid, "E", "B") - outer(Pa, Aa, Pb, Ab, mid, "B", "E")

    first = {"0": T, "+": plus, "-": minus}
    ij, pq = (Pi, Ai, Pj, Aj, Ag), (Pp, Ap, Pq, Aq, Ah)
    return first[spin[0]](*ij) * first[spin[1]](*pq)


def sigma_ll(sqrt_ps, coupling, spec_ij, spec_pq, arrays_ij=None, arrays_pq=None,
             A_norm=1.0, sum_arrays=None):
    """Cov(C^{spec_ij}, C^{spec_pq}) as an (nl, nl) matrix, l = 2 .. lmax - 1.

    `coupling` maps the names in KERNEL_FOR to (nl, nl) kernels. Array pairs default to the
    first array; `sum_arrays` (the intermediate arrays) defaults to all of them.
    """
    first = sqrt_ps.arrays[0]
    arrays_ij = arrays_ij or (first, first)
    arrays_pq = arrays_pq or (first, first)
    sum_arrays = sum_arrays or sqrt_ps.arrays
    (Pi, Pj), (Pp, Pq) = spec_ij, spec_pq
    (Ai, Aj), (Ap, Aq) = arrays_ij, arrays_pq

    # Wick pairings (i,p)(j,q) and (i,q)(j,p)
    pairings = ((Pi + Pp, Pj + Pq, (Ai, Ap), (Aj, Aq)),
                (Pi + Pq, Pj + Pp, (Ai, Aq), (Aj, Ap)))
    out = 0.0
    for s1, s2, pair1, pair2 in pairings:
        for Ag in sum_arrays:
            for Ah in sum_arrays:
                for spin in SPINS:
                    out = out + (ps_product(sqrt_ps, spin, s1, s2, pair1, pair2, Ag, Ah)
                                 * coupling[KERNEL_FOR[spin]])
    return out * A_norm ** 4
