from pspipe_utils import transfer_function as tf_tools
from pspipe_utils import consistency, log
from pspy import so_spectra, pspy_utils
from pspy import so_dict
import numpy as np
import yaml
import sys
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# log tf infos from tf yaml file
with open(d['tf_yaml'], "r") as f:
    tf_dict: dict = yaml.safe_load(f)
tf_infos = tf_dict['compute_tf.py']

spec_dir = d['spec_dir']
cov_dir = d['cov_dir']
bestfir_dir = d['bestfits_dir']

tf_dir = d['tf_dir']

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"] if d["cov_T_E_only"] else spectra

binning_file = d["binning_file"]
lmax = d["lmax"]

subtract_bf_fg = tf_infos['subtract_bf_fg']
montecarlo_errors = tf_infos['montecarlo_errors']

combins = ["AxA_AxB", "AxB_BxB"]

for sv in tf_infos["surveys_to_calib"]:
    for combin in combins:

        output_dir = f"{tf_dir}/{combin}"
        if subtract_bf_fg:
            output_dir += "_fg_sub"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        lb_list, tf_list, tf_err_list = [], [], []
        
        sv_arrays = [f'{sv}_{ar}' for ar in d[f"arrays_{sv}"]]
        for i, sv_ar in enumerate(sv_arrays):

            ref_ar = tf_infos[f"ref_map_sets"][sv_ar]

            if combin == "AxA_AxB":
                ar1A, ar2A = sv_ar, sv_ar
                ar1B, ar2B = sv_ar, ref_ar
                op = "aa/ab"
            elif combin == "AxB_BxB":
                ar1A, ar2A = sv_ar, ref_ar
                ar1B, ar2B = ref_ar, ref_ar
                op = "ab/bb"

            ar_list = [sv_ar, ref_ar]
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
                    _, bf_fg_TT_binned = pspy_utils.naive_binning(l_fg, bf_fg["TT"], binning_file, lmax)
                    ps_dict[ms1, ms2, spec] -= bf_fg_TT_binned

            lb, tf, tf_cov, _, _ = consistency.compare_spectra(ar_list, op, ps_dict, cov_dict, mode = "TT")

            tf_list.append(tf)
            tf_err_list.append(np.sqrt(tf_cov.diagonal()))
            lb_list.append(lb)

            np.savetxt(f"{output_dir}/tf_estimator_{sv_ar}.dat", np.array([lb, tf, np.sqrt(tf_cov.diagonal())]).T)
            np.savetxt(f"{output_dir}/tf_cov_{sv_ar}.dat", tf_cov)


        tf_tools.plot_tf(lb_list, tf_list, tf_err_list, d[f"arrays_{sv}"], f"{output_dir}/tf_estimation.png")
