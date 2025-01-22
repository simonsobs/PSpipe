import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pspy import so_dict, pspy_utils
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

planck_version = d["planck_version"]
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "18"

plot_dir = "paper_plot"
pspy_utils.create_directory(plot_dir)

combins = ["AxA_AxP", "AxP_PxP"]

for combin in combins:

    tf_dir = f"tf_estimator_{combin}_fg_sub"

    plt.figure(figsize=(16, 8))
    colors = ["red", "orange", "green", "blue", "gray"]
    arrays = ["pa5_f090", "pa6_f090", "pa5_f150", "pa6_f150", "pa4_f220"]

    count = 0
    for col, ar in zip(colors, arrays):
        lb, tf, tferr = np.loadtxt(f"{tf_dir}/tf_estimator_dr6_{ar}.dat", unpack=True)
        plt.errorbar(lb-10 +count*5, tf, tferr, fmt="o", color=col, label=ar)
        plt.plot(lb, lb*0+1, "--", color="black", alpha=0.5)
        count += 1
    
    na, nb = combin.split("_")
    plt.ylabel(r"$T_{\ell} = C_{\ell, \rm %s}^{\rm TT} / C_{\ell, \rm %s}^{\rm TT}$" % (na, nb), fontsize=25)
    plt.xlabel(r"$\ell$", fontsize=25)
    plt.xlim(0, 1800)
    plt.ylim(0.9, 1.05)
    plt.legend()
    plt.savefig(f"{plot_dir}/{tf_dir}_{planck_version}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()
