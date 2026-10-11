# Control variates for the pseudo-Cl covariance

`alt = sqrt(C_l) map2alm(W n_p)` is a cheap simulation whose covariance is known analytically.
`real = map2alm[W alm2map(sqrt(C_l) n_lm)]` is the expensive one, built from the same noise.
`cov(real) - c cov(alt) + c Sigma_alt` is unbiased for any `c`, with a smaller scatter when the two are correlated.

| file | |
|---|---|
| `cv_sims.py` | the two branches |
| `cv_covariance.py` | `Sigma_alt` for any number of arrays |
| `cv_spin2_kernels.py` | coupling kernels with spin-weighted 3j symbols |
| `test_control_variate.py` | tests, about 15 s |

`Sigma_alt` must be exact, because it is added back. pspy's `cov_coupling_spin0and2_simple` with its default
`planck=False` uses the spin-0 kernel for every spin structure, which is too high for polarization at low l;
`planck=True` or `cv_spin2_kernels` give the spin-weighted kernels.

Example: `notebooks/control_variate_example.ipynb`.
