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

covariances_type = ["analytic_cov", "analytic_cov_with_mc_corrections"]

ps_template = spec_dir + "/Dl_{}x{}_cross.dat"
cov_dict = {}
for cov_type in covariances_type:
    cov_template = f"{cov_dir}/{cov_type}" + "_{}x{}_{}x{}.npy"
    ps_dict, cov = consistency.get_ps_and_cov_dict(arrays, ps_template, cov_template)
    cov_dict[cov_type] = cov

lb = ps_dict["ell"]

multipole_range = {90: {"T": [800, 7000],
                        "E": [300, 7000]},
                   150: {"T": [1250, 7000],
                         "E": [300, 7000]},
                   220: {"T": [1500, 7000],
                         "E": [300, 7000]}}

operations = {"diff": "ab-cd",
              "ratio": "ab/cd"}


for i, (ar1, ar2) in enumerate(cwr(arrays, 2)):
    for j, (ar3, ar4) in enumerate(cwr(arrays, 2)):

        if j <= i: continue
        f1, f2 = d[f"freq_info_{ar1}"]["freq_tag"], d[f"freq_info_{ar2}"]["freq_tag"]
        f3, f4 = d[f"freq_info_{ar3}"]["freq_tag"], d[f"freq_info_{ar4}"]["freq_tag"]
        if f1 != f3 or f2 != f4: continue

        ar_list = [ar1, ar2, ar3, ar4]

        for m in modes:

            m0, m1 = m[0], m[1]
            lmin0, lmax0 = multipole_range[f1][m0]
            lmin1, lmax1 = multipole_range[f2][m1]
            lmin = max(lmin0, lmin1)
            lmax = min(lmax0, lmax1)


            for op_label, op in operations.items():

                if (m in ["TE", "ET"]) & (op_label == "ratio"): continue
                if ((ar1 != ar2) or (ar3 != ar4)) & (op_label == "ratio"): continue

                res_cov_dict = {}
                for cov_type in covariances_type:
                    lb, res_ps, res_cov, _, _ = consistency.compare_spectra(ar_list, op, ps_dict, cov_dict[cov_type], mode = m)

                    res_cov_dict[cov_type] = res_cov

                if op_label == "diff":
                    plot_title = f"{ar1}x{ar2} - {ar3}x{ar4}"
                    expected_res = 0.
                elif op_label == "ratio":
                    plot_title = f"{ar1}x{ar2} / {ar3}x{ar4}"
                    expected_res = 1.0

                lrange = np.where((lb >= lmin) & (lb <= lmax))[0]
                consistency.plot_residual(lb, res_ps, res_cov_dict, mode = m,
                                          title = plot_title,
                                          file_name = f"{output_dir}/{op_label}_{ar1}x{ar2}_{ar3}x{ar4}_{m}",
                                          expected_res = expected_res,
                                          lrange = lrange)
