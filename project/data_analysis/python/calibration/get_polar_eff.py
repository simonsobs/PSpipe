from pspy import pspy_utils, so_dict, so_spectra, so_cov
from pspipe_utils import consistency
import numpy as np
import pickle
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "spectra"
cov_dir = "covariances"

_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
n_bins = len(lb)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

# Define the projection pattern - i.e. which
# spectra combination will be used to compute
# the residuals
proj_pattern = np.array([1, 0, -1])

# Mode to use to get the
# polarization efficiency
used_mode = "EE"

# Create output dirs
output_dir = f"polar_efficiency_results_{used_mode}"
pspy_utils.create_directory(output_dir)

residual_output_dir = f"{output_dir}/residuals"
pspy_utils.create_directory(residual_output_dir)

chains_dir = f"{output_dir}/chains"
pspy_utils.create_directory(chains_dir)

# Define the multipole range used to obtain
# the polarization efficiencies
multipole_range = {"dr6_pa4_f150": [300, 3000],
                   "dr6_pa4_f220": [300, 3000],
                   "dr6_pa5_f090": [300, 3000],
                   "dr6_pa5_f150": [300, 3000],
                   "dr6_pa6_f090": [300, 3000],
                   "dr6_pa6_f150": [300, 3000]}

# Define the reference arrays
ref_arrays = {"dr6_pa4_f150": "dr6_pa6_f150",
              "dr6_pa4_f220": "dr6_pa4_f220",
              "dr6_pa5_f090": "dr6_pa6_f090",
              "dr6_pa5_f150": "dr6_pa6_f150",
              "dr6_pa6_f090": "dr6_pa6_f090",
              "dr6_pa6_f150": "dr6_pa6_f150"}

results_dict = {}
for ar in d["arrays_dr6"]:
    array = f"dr6_{ar}"
    if array == ref_arrays[array]: continue
    ref_array = ref_arrays[array]

    spectra_order = [(array, array, used_mode),
                     (array, ref_array, used_mode),
                     (ref_array, ref_array, used_mode)]

    # Load spectra and cov
    ps_dict = {}
    cov_dict = {}
    for i, (ar1, ar2, m1) in enumerate(spectra_order):
        _, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{ar1}x{ar2}_cross.dat",
                                   spectra = spectra)
        ps_dict[ar1, ar2, m1] = ps[m1]
        for j, (ar3, ar4, m2) in enumerate(spectra_order):
            if j < i: continue
            cov = np.load(f"{cov_dir}/analytic_cov_{ar1}x{ar2}_{ar3}x{ar4}.npy")
            cov = so_cov.selectblock(cov, modes, n_bins = n_bins,
                                     block = used_mode + used_mode)
            cov_dict[(ar1, ar2, m1), (ar3, ar4, m2)] = cov

    # Concatenation
    spec_vec, full_cov = consistency.append_spectra_and_cov(ps_dict, cov_dict,
                                                               spectra_order)

    # Save and plot residuals before calibration
    res_spectrum, res_cov = consistency.project_spectra_vec_and_cov(spec_vec, full_cov, proj_pattern)
    np.savetxt(f"{residual_output_dir}/residual_{array}_before.dat", np.array([lb, res_spectrum]).T)
    np.savetxt(f"{residual_output_dir}/residual_cov_{array}.dat", res_cov)

    lmin, lmax = multipole_range[array]
    id = np.where((lb >= lmin) & (lb <= lmax))[0]
    consistency.plot_residual(lb[id], res_spectrum[id], res_cov[np.ix_(id, id)],
                                 used_mode, array, f"{residual_output_dir}/residual_{array}_before")

    cal_mean, cal_std = consistency.get_calibration_amplitudes(spec_vec, full_cov,
                                                                proj_pattern, used_mode,
                                                                id, f"{chains_dir}/{array}")
    results_dict[array] = {"multipole_range": multipole_range[array],
                           "ref_array": ref_arrays[array],
                           "calibs": [cal_mean, cal_std]}

    if used_mode == "EE":
        calib_vec = np.array([cal_mean**2, cal_mean, 1])
    elif used_mode == "TE":
        calib_vec = np.array([cal_mean, 1, 1])

    res_spectrum, res_cov = consistency.project_spectra_vec_and_cov(spec_vec, full_cov,
                                                                      proj_pattern, calib_vec = calib_vec)
    np.savetxt(f"{residual_output_dir}/residual_{array}_after.dat", np.array([lb, res_spectrum]).T)
    consistency.plot_residual(lb[id], res_spectrum[id], res_cov[np.ix_(id, id)],
                                 used_mode, array, f"{residual_output_dir}/residual_{array}_after")

pickle.dump(results_dict, open(f"{output_dir}/polareff_dict.pkl", "wb"))
