import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pspy import so_dict, pspy_utils
import sys
import pickle

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
planck_version = d["planck_version"]

labelsize = 14
fontsize = 20
recalibrate = True

cal_dir = "calibration_results_planck_bias_corrected_fg_sub"
plot_dir = "plots/paper_plot"
pspy_utils.create_directory(plot_dir)

combins = ["AxA_AxP", "AxP_PxP"]
cal_dict_combin = ["AxA-AxP", "PxP-AxP"] # we use different name, should be fixed at some point

with open(f"{cal_dir}/calibs_dict.pkl", "rb") as f:
    cal_dict = pickle.load(f)

for c_combin, combin in zip(cal_dict_combin, combins):
    print(c_combin)
    print(combin)

    tf_dir = f"tf_estimator_{combin}_fg_sub"

    plt.figure(figsize=(12, 6), dpi=100)
    colors = ["red", "orange", "green", "blue", "gray"]
    arrays = ["pa5_f090", "pa6_f090", "pa5_f150", "pa6_f150", "pa4_f220"]

    count = 0
    for col, ar in zip(colors, arrays):
        lb, tf, tferr = np.loadtxt(f"{tf_dir}/tf_estimator_dr6_{ar}.dat", unpack=True)
        if recalibrate:
            cal, sigma = cal_dict[c_combin, ar]["calibs"]
            print(ar, cal)
            tf, tferr = tf * cal, tferr * cal
        
        pa, freq = ar.split("_")
        
        if ar == "pa4_f220":
            plt.errorbar(lb-14 +count*7, tf, tferr, fmt="o", color=col, label=f"{pa.upper()} {freq}", alpha=0.6, mfc='w')
        else:
            plt.errorbar(lb-14 +count*7, tf, tferr, fmt="o", color=col, label=f"{pa.upper()} {freq}", mfc='w')

        plt.plot(lb, lb*0+1, "--", color="black", alpha=0.5)
        count += 1
    
    na, nb = combin.split("_")
    plt.ylabel(r"$T_{\ell} = C_{\ell, \rm %s}^{\rm TT} / C_{\ell, \rm %s}^{\rm TT}$" % (na, nb), fontsize=fontsize)
    plt.xlabel(r"$\ell$", fontsize=fontsize)
    plt.xlim(0, 1800)
    plt.ylim(0.9, 1.05)
    plt.legend(fontsize=labelsize)
    plt.tick_params(labelsize=labelsize)
   # plt.show()
    plt.savefig(f"{plot_dir}/{tf_dir}_{planck_version}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()
