import numpy as np
from pspy import so_dict
import sys
from pspipe_utils import transfer_function as tf_tools

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

combin = "AxA:AxP"
#combin = "AxP:PxP"

subtract_bf_fg = True
method = "logistic"

output_dir = f"tf_estimator_{combin}"

if subtract_bf_fg:
    output_dir += "_fg_sub"

sv = "dr6"
arrays = [f"{sv}_{ar}" for ar in d[f"arrays_{sv}"]]


prior_dict = {}

prior_dict["aa", "logistic"] = {"min": 0, "max": 5}
prior_dict["bb", "logistic"] = {"min": 0, "max": 5}
prior_dict["cc", "logistic"] = {"min": 0, "max": 1}

prior_dict["aa", "beta"] = {"min": 0, "max": 1}
prior_dict["bb", "beta"] = {"min": 100, "max": 1500}
prior_dict["cc", "beta"] = {"min": 0, "max": 5}

prior_dict["aa", "sigurd2"] = {"min": 0, "max": 5}
prior_dict["bb", "sigurd2"] = {"min": 0, "max": 5}
prior_dict["cc", "sigurd2"] = {"min": 30, "max": 850}

method = "logistic"
fixed_amp = True
suffix = "_fixed_amp" if fixed_amp else ""

lb_list = []
tf_list = []
tf_err_list = []
tf_model_list = []
ell_list = []

lmax_plot = {"dr6_pa4_f150": 2000,
             "dr6_pa4_f220": 2500,
             "dr6_pa5_f090": 1200,
             "dr6_pa5_f150": 2000,
             "dr6_pa6_f090": 1200,
             "dr6_pa6_f150": 2000}

pars = ["bb", "cc"] if fixed_amp else ["aa", "bb", "cc"]

for ar in arrays:

    lb, tf_est, tf_err = np.loadtxt(f"{output_dir}/tf_estimator_{ar}.dat").T
    tf_cov = np.loadtxt(f"{output_dir}/tf_cov_{ar}.dat")

    id = np.where(lb < lmax_plot[ar])[0]
    lb, tf_est, tf_err, tf_cov = lb[id], tf_est[id], tf_err[id], tf_cov[np.ix_(id, id)]

    chain_name = f"{output_dir}/chains_tf/{ar}"
    tf_tools.fit_tf(lb, tf_est, tf_cov, prior_dict, chain_name, method=method, fixed_amp=fixed_amp)

    mus, stds = tf_tools.get_parameter_mean_and_std(chain_name, pars)

    if fixed_amp:
        aa = 1.
        bb, cc = mus
    else:
        aa, bb, cc = mus

    ell = np.arange(2, d["lmax"])
    bestfit_tf = tf_tools.tf_model(ell, aa, bb, cc, method=method)
    np.savetxt(f"{output_dir}/tf_fit_{ar}_{method}{suffix}.dat", np.array([ell, bestfit_tf]).T)

    id_model = np.where(ell <= lmax_plot[ar])[0]
    ell, bestfit_tf = ell[id_model], bestfit_tf[id_model]

    lb_list.append(lb)
    tf_list.append(tf_est)
    tf_err_list.append(tf_err)
    tf_model_list.append(bestfit_tf)
    ell_list.append(ell)


tf_tools.plot_tf(lb_list, tf_list, tf_err_list, arrays,
                 f"{output_dir}/tf_estimation_fit_{method}{suffix}.png",
                 ell_list=ell_list, tf_model_list=tf_model_list)
