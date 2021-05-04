import matplotlib
from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys, os
from itertools import combinations_with_replacement as cwr
from itertools import combinations

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
n_surveys = len(surveys)
arrays = [d["arrays_%s" % survey] for survey in surveys]

spec_dir = "../../spectra/"
cov_dir = "../../covariances/"

output_dir = "../outputs/"
output_plot_dir = os.path.join(output_dir, "plots/")
output_data_dir = os.path.join(output_dir, "data/")
pspy_utils.create_directory(output_plot_dir)
pspy_utils.create_directory(output_data_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

#### SEASON NULL TESTS ####
if n_surveys < 2:
    sys.exit("Cannot make season null tests with only 1 season.")

spectra_names = [[m for m in cwr(arrays[i], 2)] for i in range(n_surveys)]
survey_differences = [m for m in combinations(surveys, 2)]
index_differences = [m for m in combinations(np.arange(n_surveys), 2)]

chi2_array = {}
chi2_dict = {}

spectra_dict = {}
residual_dict = {}

for index, (surv1, surv2) in enumerate(survey_differences):

    chi2_array[surv1, surv2] = []
    for (spec1, spec2) in spectra_names[index_differences[index][0]]:

        if not (spec1, spec2) in spectra_names[index_differences[index][1]]:
            continue

        else:
            line = []
            ndof_line = []

            lb, ps1 = so_spectra.read_ps(
                          spec_dir + "Dl_%s_%sx%s_%s_cross.dat" % (
                          surv1, spec1, surv1, spec2), spectra = spectra)

            lb, ps2 = so_spectra.read_ps(
                          spec_dir + "Dl_%s_%sx%s_%s_cross.dat" % (
                          surv2, spec1, surv2, spec2), spectra = spectra)

            cov11 = np.load(cov_dir + "analytic_cov_%s_%sx%s_%s_%s_%sx%s_%s.npy" % (
                        surv1, spec1, surv1, spec2, surv1, spec1, surv1, spec2))

            cov22 = np.load(cov_dir + "analytic_cov_%s_%sx%s_%s_%s_%sx%s_%s.npy" % (
                        surv2, spec1, surv2, spec2, surv2, spec1, surv2, spec2))

            cov12 = np.load(cov_dir + "analytic_cov_%s_%sx%s_%s_%s_%sx%s_%s.npy" % (
                        surv1, spec1, surv1, spec2, surv2, spec1, surv2, spec2))

            # Compute residual covariance matrix
            res_cov = cov11 + cov22 - 2 * cov12

            # TT+TE+EE chi2
            ps1_ttteee = np.concatenate([ps1[mode] for mode in modes])
            ps2_ttteee = np.concatenate([ps2[mode] for mode in modes])
            residual_ps = ps1_ttteee - ps2_ttteee
            chi2 = residual_ps @ np.linalg.solve(res_cov, residual_ps)
            chi2_dict["all", surv1, surv2, spec1, spec2] = [chi2, len(residual_ps)]
            line.append(chi2)
            ndof_line.append(len(residual_ps))

            # Loop on TT, TE, ET, EE
            for mode in modes:

                # Select block
                res_covblock = so_cov.selectblock(res_cov, modes,
                                                 n_bins = len(lb),
                                                 block = mode + mode)

                # Compute power spectrum and errors
                res_std = np.sqrt(res_covblock.diagonal())
                res_ps = ps1[mode] - ps2[mode]

                # Compute chi2
                chi2 = res_ps @ np.linalg.solve(res_covblock, res_ps)
                chi2_dict[mode, surv1, surv2, spec1, spec2] = [chi2, len(lb)]
                line.append(chi2)
                ndof_line.append(len(lb))

                # Low-ell (ell < 1000) chi2
                id_low = np.where(lb < 1000)
                low_ps = res_ps[id_low]
                low_covblock = res_covblock[np.min(id_low):np.max(id_low)+1,
                                            np.min(id_low):np.max(id_low)+1]
                chi2_low = low_ps @ np.linalg.solve(low_covblock, low_ps)
                chi2_dict[mode, surv1, surv2, spec1, spec2, "low"] = [chi2_low, len(id_low[0])]
                line.append(chi2_low)
                ndof_line.append(len(id_low[0]))

                # High-ell (ell > 1000) chi2
                id_high = np.where(lb > 1000)
                high_ps = res_ps[id_high]
                high_covblock = res_covblock[np.min(id_high):np.max(id_high)+1,
                                             np.min(id_high):np.max(id_high)+1]
                chi2_high = high_ps @ np.linalg.solve(high_covblock, high_ps)
                chi2_dict[mode, surv1, surv2, spec1, spec2, "high"] = [chi2_high, len(id_high[0])]
                line.append(chi2_high)
                ndof_line.append(len(id_high[0]))

                # Spectra dict
                covblock1 = so_cov.selectblock(cov11, modes,
                                              n_bins = len(lb),
                                              block = mode + mode)
                covblock2 = so_cov.selectblock(cov22, modes,
                                              n_bins = len(lb),
                                              block = mode + mode)
                std1 = np.sqrt(covblock1.diagonal())
                std2 = np.sqrt(covblock2.diagonal())

                spectra_dict[mode, surv1, spec1, spec2] = [lb, ps1[mode], std1]
                spectra_dict[mode, surv2, spec1, spec2] = [lb, ps2[mode], std2]

                # Residual dict
                residual_dict[mode, surv1, surv2, spec1, spec2] = [lb, res_ps, res_std]

            chi2_array[surv1, surv2].append(line)

# Save dicts
pickle.dump(chi2_array, open(os.path.join(output_data_dir, "chi2_arrays.pkl"), "wb"))
pickle.dump(spectra_dict, open(os.path.join(output_data_dir, "spectra.pkl"), "wb"))
pickle.dump(residual_dict, open(os.path.join(output_data_dir, "residual.pkl"), "wb"))

# Histogram plots
labels = np.array(["TT+TE+EE", "TT", r"TT ($\ell$ $\leq$ 1000)", r"TT ($\ell$ $\geq$ 1000)",
                       "TE", r"TE ($\ell$ $\leq$ 1000)", r"TE ($\ell$ $\geq$ 1000)",
                       "ET", r"ET ($\ell$ $\leq$ 1000)", r"ET ($\ell$ $\geq$ 1000)",
                       "EE", r"EE ($\ell$ $\leq$ 1000)", r"EE ($\ell$ $\geq$ 1000)",
                       "None", "None"])
labels = labels.reshape(3, 5)

tot_chi2_array = []

for index, (surv1, surv2) in enumerate(survey_differences):

    chi2_array_s1s2 = np.array(chi2_array[surv1, surv2], dtype = np.float)
    tot_chi2_array.append(chi2_array_s1s2)

    fig, axes = plt.subplots(3, 5, figsize = (15, 9))

    for i in range(len(chi2_array_s1s2[0])):
        axes[i//5, i%5].hist(chi2_array_s1s2[:, i], label = labels[i//5, i%5])
        axes[i//5, i%5].axvline(ndof_line[i], ymin = -1, ymax = +2,
                                color = "k", ls = "--")
        axes[i//5, i%5].legend(frameon=False)
    axes[2, -1].axis('off')
    axes[2, -2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir + "chi2_season_null_%s_%s.pdf" % (surv1, surv2)))

tot_chi2_array = np.vstack(tot_chi2_array)
fig, axes = plt.subplots(3, 5, figsize = (15, 9))
for i in range(len(tot_chi2_array[0])):
    axes[i//5, i%5].hist(tot_chi2_array[:, i], label = labels[i//5, i%5])
    axes[i//5, i%5].axvline(ndof_line[i], ymin = -1, ymax = +2,
                            color = "k", ls = "--")
    axes[i//5, i%5].legend(frameon = False)
axes[2, -1].axis('off')
axes[2, -2].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_plot_dir, "chi2_season_null.pdf"))
