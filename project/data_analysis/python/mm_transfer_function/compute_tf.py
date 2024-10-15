from pspipe_utils import transfer_function as tf_tools
from pspipe_utils import consistency, log
from pspy import so_spectra, pspy_utils
from pspy import so_dict
import numpy as np
import sys
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spec_dir = "spectra_leak_corr_planck_bias_corr"
cov_dir = "covariances"
bestfir_dir = "best_fits"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"] if d["cov_T_E_only"] else spectra

sv = "dr6"
arrays = [f"{sv}_{ar}" for ar in d[f"arrays_{sv}"]]
binning_file = d["binning_file"]
lmax = d["lmax"]

ref_ars = {"dr6_pa4_f150": "Planck_f143",
           "dr6_pa4_f220": "Planck_f217",
           "dr6_pa5_f090": "Planck_f143",
           "dr6_pa5_f150": "Planck_f143",
           "dr6_pa6_f090": "Planck_f143",
           "dr6_pa6_f150": "Planck_f143"}

subtract_bf_fg = True
montecarlo_errors = True

combins = ["AxA_AxP", "AxP_PxP"]

for combin in combins:

    output_dir = f"tf_estimator_{combin}"
    if subtract_bf_fg:
        output_dir += "_fg_sub"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lb_list, tf_list, tf_err_list = [], [], []
    
    for i, ar in enumerate(arrays):

        ref_ar = ref_ars[ar]

        if combin == "AxA_AxP":
            ar1A, ar2A = ar, ar
            ar1B, ar2B = ar, ref_ar
            op = "aa/ab"
        elif combin == "AxP_PxP":
            ar1A, ar2A = ar, ref_ar
            ar1B, ar2B = ref_ar, ref_ar
            op = "ab/bb"

        ar_list = [ar, ref_ar]
        ps_template = spec_dir + "/Dl_{}x{}_cross.dat"
        
        if montecarlo_errors == True:
            cov_template = f"{cov_dir}/mc_cov" + "_{}x{}_{}x{}.npy"
        else:
            cov_template = f"{cov_dir}/analytic_cov" + "_{}x{}_{}x{}.npy"

        ps_dict, cov_dict = consistency.get_ps_and_cov_dict(ar_list, ps_template, cov_template)
 
        if subtract_bf_fg:
            for ms1, ms2, spec in ps_dict.keys():
                if spec != "TT":
                    continue
                log.info(f"remove fg {spec}  {ms1} x {ms2}")
                l_fg, bf_fg = so_spectra.read_ps(f"{bestfir_dir}/fg_{ms1}x{ms2}.dat", spectra=spectra)
                _, bf_fg_TT_binned = pspy_utils.naive_binning(l_fg, bf_fg["TT"], d["binning_file"], d["lmax"])
                ps_dict[ms1, ms2, spec] -= bf_fg_TT_binned

        lb, tf, tf_cov, _, _ = consistency.compare_spectra(ar_list, op, ps_dict, cov_dict, mode = "TT")

        tf_list.append(tf)
        tf_err_list.append(np.sqrt(tf_cov.diagonal()))
        lb_list.append(lb)


        np.savetxt(f"{output_dir}/tf_estimator_{ar}.dat", np.array([lb, tf, np.sqrt(tf_cov.diagonal())]).T)
        np.savetxt(f"{output_dir}/tf_cov_{ar}.dat", tf_cov)


    tf_tools.plot_tf(lb_list, tf_list, tf_err_list, arrays, f"{output_dir}/tf_estimation.png")
