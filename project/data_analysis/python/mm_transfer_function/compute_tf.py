from pspipe_utils import transfer_function as tf_tools
from pspy import so_spectra, so_cov
from pspy import so_dict
import numpy as np
import sys
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "spectra"
cov_dir = "covariances"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

sv = "dr6"
arrays = [f"{sv}_{ar}" for ar in d[f"arrays_{sv}"]]

combin = "AxA:AxP"
#combin = "AxP:PxP"

output_dir = f"tf_estimator_{combin}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

downgrade = 3
binning_file = d["binning_file"]
lmax = d["lmax"]

ref_ars = {"dr6_pa4_f150": "Planck_f143",
           "dr6_pa4_f220": "Planck_f217",
           "dr6_pa5_f090": "Planck_f100",
           "dr6_pa5_f150": "Planck_f143",
           "dr6_pa6_f090": "Planck_f100",
           "dr6_pa6_f150": "Planck_f143"}


lb_list = []
tf_list = []
tf_err_list = []

for i, ar in enumerate(arrays):

    ref_ar = ref_ars[ar]

    if combin == "AxA:AxP":
        ar1A, ar2A = ar, ar
        ar1B, ar2B = ar, ref_ar
    elif combin == "AxP:PxP":
        ar1A, ar2A = ar, ref_ar
        ar1B, ar2B = ref_ar, ref_ar

    lb, psA = so_spectra.read_ps(f"{spec_dir}/Dl_{ar1A}x{ar2A}_cross.dat", spectra = spectra)
    lb, psB = so_spectra.read_ps(f"{spec_dir}/Dl_{ar1B}x{ar2B}_cross.dat", spectra = spectra)

    covAA = np.load(f"{cov_dir}/analytic_cov_{ar1A}x{ar2A}_{ar1A}x{ar2A}.npy")
    covAB = np.load(f"{cov_dir}/analytic_cov_{ar1A}x{ar2A}_{ar1B}x{ar2B}.npy")
    covBB = np.load(f"{cov_dir}/analytic_cov_{ar1B}x{ar2B}_{ar1B}x{ar2B}.npy")

    covAA = so_cov.selectblock(covAA, modes,
                               n_bins = len(lb),
                               block = "TTTT")
    covAB = so_cov.selectblock(covAB, modes,
                               n_bins = len(lb),
                               block = "TTTT")
    covBB = so_cov.selectblock(covBB, modes,
                               n_bins = len(lb),
                               block = "TTTT")


    lb, ttA, covAA = tf_tools.downgrade_binning(psA["TT"], covAA, downgrade, binning_file, lmax)
    lb, ttB, covBB = tf_tools.downgrade_binning(psB["TT"], covBB, downgrade, binning_file, lmax)
    lb, _, covAB = tf_tools.downgrade_binning(psA["TT"], covAB, downgrade, binning_file, lmax)

    snr = ttB / np.sqrt(covBB.diagonal())
    if ar == "dr6_pa4_f220":
        id = np.where(snr > 3)
    else:
        id = np.where((snr > 3) & (lb <= 2000))

    ttA, ttB = ttA[id], ttB[id]
    covAA = covAA[np.ix_(id[0], id[0])]
    covAB = covAB[np.ix_(id[0], id[0])]
    covBB = covBB[np.ix_(id[0], id[0])]

    lb = lb[id]

    tf = tf_tools.get_tf_unbiased_estimator(ttA, ttB, covAB, covBB)
    tf_cov = tf_tools.get_tf_estimator_covariance(ttA, ttB, covAA, covAB, covBB)

    tf_list.append(tf)
    tf_err_list.append(np.sqrt(tf_cov.diagonal()))
    lb_list.append(lb)

    np.savetxt(f"{output_dir}/tf_estimator_{ar}.dat", np.array([lb, tf, np.sqrt(tf_cov.diagonal())]).T)
    np.savetxt(f"{output_dir}/tf_cov_{ar}.dat", tf_cov)


tf_tools.plot_tf(lb_list, tf_list, tf_err_list,
                 arrays, f"{output_dir}/tf_estimation.png")
