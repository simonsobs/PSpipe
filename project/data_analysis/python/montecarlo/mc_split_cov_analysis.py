"""
This script analyzes a set of simulated spectra by
computing their montecarlo covariance and writing
MC corrected covariances to disk
"""


from pspy import pspy_utils, so_dict, so_spectra
from pspipe_utils import covariance
import numpy as np
import sys
from itertools import combinations_with_replacement as cwr
from itertools import combinations

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
iStart = d["iStart"]
iStop = d["iStop"]

sim_spec_dir = d["sim_spec_dir"]
cov_dir = "split_covariances"

pspy_utils.create_directory(cov_dir)

only_diag_corrections = True
if only_diag_corrections:
    corrected_cov_name = "analytic_cov_with_diag_mc_corrections"
else:
    corrected_cov_name = "analytic_cov_with_mc_corrections"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}

spec_list = []
import os
for sv in surveys:

    for ar in arrays[sv]:

        ns = len(d[f"maps_{sv}_{ar}"])

        cross_split_list = list(combinations(np.arange(ns, dtype = np.int32), 2))

        for s1, s2 in cross_split_list:
            spec_list.append(f"{sv}_{ar}_{s1}x{sv}_{ar}_{s2}")

        cross_spec_list = list(cwr(cross_split_list, 2))

        for (s1, s2), (s3, s4) in cross_spec_list:
            #if os.path.exists(f"{cov_dir}/mc_cov_{sv}_{ar}_{s1}x{sv}_{ar}_{s2}_{sv}_{ar}_{s3}x{sv}_{ar}_{s4}.npy"):
                #continue

            ps_list_12 = []
            ps_list_34 = []
            for iii in range(iStart, iStop + 1):
                spec_name_cross_12 = f"{type}_{sv}_{ar}x{sv}_{ar}_{s1}{s2}_{iii:05d}.dat"
                spec_name_cross_34 = f"{type}_{sv}_{ar}x{sv}_{ar}_{s3}{s4}_{iii:05d}.dat"

                lb, ps_12 = so_spectra.read_ps(f"{sim_spec_dir}/{spec_name_cross_12}", spectra = spectra)
                lb, ps_34 = so_spectra.read_ps(f"{sim_spec_dir}/{spec_name_cross_34}", spectra = spectra)

                vec_12 = []
                vec_34 = []
                for spec in spectra:
                    vec_12 = np.append(vec_12, ps_12[spec])
                    vec_34 = np.append(vec_34, ps_34[spec])

                ps_list_12 += [vec_12]
                ps_list_34 += [vec_34]

            cov_mc = 0
            for iii in range(iStart, iStop + 1):
                cov_mc += np.outer(ps_list_12[iii], ps_list_34[iii])

            cov_mc = cov_mc / (iStop + 1 - iStart) - np.outer(np.mean(ps_list_12, axis = 0), np.mean(ps_list_34, axis = 0))

            np.save(f"{cov_dir}/mc_cov_{sv}_{ar}_{s1}x{sv}_{ar}_{s2}_{sv}_{ar}_{s3}x{sv}_{ar}_{s4}.npy", cov_mc)

lb, _ = so_spectra.read_ps(f"{sim_spec_dir}/{type}_{sv}_{ar}x{sv}_{ar}_{s1}{s2}_00000.dat")
n_bins = len(lb)

for ar in arrays["dr6"]:
    spec_list_ar = [s for s in spec_list if ar in s]
    mc_full_cov = covariance.read_cov_block_and_build_full_cov(spec_list_ar, cov_dir, cov_type = "mc_cov", spectra_order=spectra)
    an_full_cov = covariance.read_cov_block_and_build_full_cov(spec_list_ar, cov_dir, cov_type = "analytic_cov", spectra_order=spectra)

    mc_corrected_full_cov = covariance.correct_analytical_cov(an_full_cov, mc_full_cov,
                                                              only_diag_corrections = only_diag_corrections)
    mc_corrected_cov_dict = covariance.full_cov_to_cov_dict(mc_corrected_full_cov, spec_list_ar, n_bins, spectra_order=spectra)

    covariance.cov_dict_to_file(mc_corrected_cov_dict, spec_list_ar, cov_dir, cov_type = corrected_cov_name,
                                spectra_order=spectra)
