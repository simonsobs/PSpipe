from pspy import so_dict, so_spectra, so_cov, pspy_utils
from itertools import combinations_with_replacement as cwr
from pspipe_utils import consistency
import matplotlib.pyplot as plt
import numpy as np
import sys

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "spectra"
cov_dir = "covariances"

output_dir = "plots/nulls"
pspy_utils.create_directory(output_dir)

surveys = d["surveys"]
arrays = []
for sv in surveys:
    for ar in d[f"arrays_{sv}"]:
        arrays.append(f"{sv}_{ar}")

ps_dict = {}
cov_dict = {}

for i, (ar1, ar2) in enumerate(cwr(arrays, 2)):

    lb, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{ar1}x{ar2}_cross.dat",
                                spectra = spectra)
    for j, (ar3, ar4) in enumerate(cwr(arrays, 2)):
        if j < i: continue
        cov = np.load(f"{cov_dir}/analytic_cov_{ar1}x{ar2}_{ar3}x{ar4}.npy")

        for m in modes:

            subcov = so_cov.selectblock(cov, modes, len(lb), block = m + m)
            ps_dict[ar1, ar2, m] = ps[m]
            cov_dict[(ar1, ar2, m), (ar3, ar4, m)] = subcov

multipole_range = {90: {"T": [800, 7000],
                        "E": [300, 7000]},
                   150: {"T": [1250, 7000],
                         "E": [300, 7000]},
                   220: {"T": [1500, 7000],
                         "E": [300, 7000]}}

for i, (ar1, ar2) in enumerate(cwr(arrays, 2)):
    for j, (ar3, ar4) in enumerate(cwr(arrays, 2)):

        if j <= i: continue
        f1, f2 = d[f"nu_eff_{ar1}"], d[f"nu_eff_{ar2}"]
        f3, f4 = d[f"nu_eff_{ar3}"], d[f"nu_eff_{ar4}"]
        if f1 != f3 or f2 != f4: continue

        for m in modes:

            m0, m1 = m[0], m[1]
            lmin0, lmax0 = multipole_range[f1][m0]
            lmin1, lmax1 = multipole_range[f2][m1]
            lmin = max(lmin0, lmin1)
            lmax = min(lmax0, lmax1)
            lrange = np.where((lb >= lmin) & (lb <= lmax))[0]

            ps_order = [(ar1, ar2, m), (ar3, ar4, m)]
            ps_vec, full_cov = consistency.append_spectra_and_cov(ps_dict, cov_dict, ps_order)
            ps_res, cov_res = consistency.project_spectra_vec_and_cov(ps_vec, full_cov, [1, -1])

            consistency.plot_residual(lb, ps_res, cov_res, m,
                                         f"{ar1}x{ar2} - {ar3}x{ar4}",
                                         f"{output_dir}/residual_{ar1}x{ar2}_{ar3}x{ar4}_{m}",
                                         lrange = lrange)
