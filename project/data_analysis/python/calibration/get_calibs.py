from pspy import pspy_utils, so_dict, so_consistency, so_spectra, so_cov
import numpy as np
import pickle
import sys


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "../spectra"
cov_dir = "../covariances"

spec_file = "Dl_%sx%s_cross.dat"
cov_file = "analytic_cov_%sx%s_%sx%s.npy"

_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
n_bins = len(lb)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

# Define the projection pattern - i.e. which
# spectra combination will be used to compute
# the residuals with
#   A: the array you want to calibrate
#   B: the reference array
spectra_combin = "AxA-AxB"

if spectra_combin == "AxA-AxB":
    proj_pattern = np.array([1, -1, 0])
    #                     [[ AxA,
    #    [[1, -1, 0]]  x     AxP,    =  AxA - AxP
    #                        PxP ]]
elif spectra_combin == "AxA-BxB":
    proj_pattern = np.array([1, 0, -1])
    #                     [[ AxA,
    #    [[1, 0, -1]]  x     AxP,    =  AxA - PxP
    #                        PxP ]]

elif spectra_combin == "BxB-AxB":
    proj_pattern = np.array([0, -1, 1])
    #                     [[ AxA,
    #    [[0, -1, 1]]  x     AxP,    =  PxP - AxP
    #                        PxP ]]

# Create output dirs
output_dir = f"calibration_results_{spectra_combin}"
pspy_utils.create_directory(output_dir)

residual_output_dir = f"{output_dir}/residuals"
pspy_utils.create_directory(residual_output_dir)

chains_dir = f"{output_dir}/chains"
pspy_utils.create_directory(chains_dir)

# Define the multipole range used to obtain
# the calibration amplitudes
multipole_range = {"dr6_pa4_f150": [1250, 1800],
                   "dr6_pa4_f220": [1500, 2000],
                   "dr6_pa5_f090": [800, 1100],
                   "dr6_pa5_f150": [1250, 1800],
                   "dr6_pa6_f090": [800, 1100],
                   "dr6_pa6_f150": [1250, 1800]}
pickle.dump(multipole_range, open(f"{output_dir}/multipole_range.pkl", "wb"))

# Define the reference arrays
ref_arrays = {"dr6_pa4_f150": "Planck_f143",
             "dr6_pa4_f220": "Planck_f217",
             "dr6_pa5_f090": "Planck_f100",
             "dr6_pa5_f150": "Planck_f143",
             "dr6_pa6_f090": "Planck_f100",
             "dr6_pa6_f150": "Planck_f143"}
pickle.dump(ref_arrays, open(f"{output_dir}/reference_arrays.pkl", "wb"))

calib_dict = {}
for ar in d["arrays_dr6"]:

    array = f"dr6_{ar}"
    ref_array = ref_arrays[array]

    spectra_order = [(array, array),
                     (array, ref_array),
                     (ref_array, ref_array)]

    # Load spectra and cov
    spectra_list = []
    cov_list = []
    for i, (ar1, ar2) in enumerate(spectra_order):
        _, ps = so_spectra.read_ps(f"{spec_dir}/{spec_file}" % (ar1, ar2),
                                   spectra = spectra)
        spectra_list.append(ps["TT"])
        for j, (ar3, ar4) in enumerate(spectra_order):
            if j < i: continue
            cov = np.load(f"{cov_dir}/{cov_file}" % (ar1, ar2, ar3, ar4))
            cov = so_cov.selectblock(cov, modes, n_bins = n_bins, block = "TTTT")
            cov_list.append(cov)

    # Concatenation
    spec_vec, full_cov = so_consistency.append_spectra_and_cov(spectra_list, cov_list)

    # Save and plot residuals before calibration
    res_spectrum, res_cov = so_consistency.project_spectra_vec_and_cov(spec_vec, full_cov, proj_pattern)
    np.savetxt(f"{residual_output_dir}/residual_{array}_before.dat", np.array([lb, res_spectrum]).T)
    np.savetxt(f"{residual_output_dir}/residual_cov_{array}.dat", res_cov)

    lmin, lmax = multipole_range[array]
    id = np.where((lb >= lmin) & (lb <= lmax))
    so_consistency.plot_residual(lb[id], res_spectrum[id], res_cov[np.ix_(id[0], id[0])],
                     "TT", array, f"{residual_output_dir}/residual_{array}_before")

    # Calibrate the spectra
    cal_mean, cal_std = so_consistency.get_calibration_amplitudes(spec_vec, full_cov,
                                                    proj_pattern, "TT", id,
                                                    f"{chains_dir}/{array}")
    calib_dict[array] = [cal_mean, cal_std]

    calib_vec = np.array([cal_mean**2, cal_mean, 1])
    res_spectrum, res_cov = so_consistency.project_spectra_vec_and_cov(spec_vec, full_cov,
                                                          proj_pattern,
                                                          calib_vec = calib_vec)
    np.savetxt(f"{residual_output_dir}/residual_{array}_after.dat", np.array([lb, res_spectrum]).T)
    so_consistency.plot_residual(lb[id], res_spectrum[id], res_cov[np.ix_(id[0], id[0])],
                     "TT", array, f"{residual_output_dir}/residual_{array}_after")

pickle.dump(calib_dict, open(f"{output_dir}/calibs_dict.pkl", "wb"))
