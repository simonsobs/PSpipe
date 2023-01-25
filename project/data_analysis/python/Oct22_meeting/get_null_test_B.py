from pspy import so_dict, so_spectra, so_cov, pspy_utils
from itertools import combinations_with_replacement as cwr
from pspipe_utils import consistency
import matplotlib.pyplot as plt
import numpy as np
import sys

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TB", "BT", "EB", "BE", "BB"]

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "spectra"
cov_dir = "covariances"
n_sims = 280
output_dir = "plots/nulls"
pspy_utils.create_directory(output_dir)


null_pairs =  [["dr6_pa4_f150xdr6_pa4_f150", "dr6_pa5_f150xdr6_pa5_f150"]]
null_pairs +=  [["dr6_pa4_f150xdr6_pa4_f150", "dr6_pa6_f150xdr6_pa6_f150"]]
null_pairs +=  [["dr6_pa5_f150xdr6_pa5_f150", "dr6_pa6_f150xdr6_pa6_f150"]]
null_pairs +=  [["dr6_pa5_f090xdr6_pa5_f090", "dr6_pa6_f090xdr6_pa6_f090"]]
print(null_pairs)

pow = {}
pow["TB"] = 1
pow["BT"] = 1
pow["EB"] = 0
pow["BE"] = 0
pow["BB"] = 0

l, Dl_TT, Dl_EE, Dl_BB, Dl_TE = np.loadtxt("cosmo2017_10K_acc3_lensedCls.dat", unpack=True)

alpha = 0.3 * np.pi / 180
overplot_theory_lines = {}
overplot_theory_lines["TB"] = None
overplot_theory_lines["BT"] = None
overplot_theory_lines["EB"] = [l, 1./2 * np.sin(4 * alpha) * (Dl_EE + Dl_BB)]
overplot_theory_lines["BE"] = [l, 1./2 * np.sin(4 * alpha) * (Dl_EE + Dl_BB)]
overplot_theory_lines["BB"] = None

for null in null_pairs:
    spec_a, spec_b = null
    
    lb, Db_a = so_spectra.read_ps(f"spectra/Dl_{spec_a}_cross.dat", spectra=spectra)
    lb, Db_b = so_spectra.read_ps(f"spectra/Dl_{spec_b}_cross.dat", spectra=spectra)

    for m in modes:
        
        diff = Db_a[m] - Db_b[m]

        diff_list = []
        for iii in range(n_sims):
            lb, Db_a_sim = so_spectra.read_ps(f"sim_spectra/Dl_{spec_a}_cross_%05d.dat" % iii, spectra=spectra)
            lb, Db_b_sim = so_spectra.read_ps(f"sim_spectra/Dl_{spec_b}_cross_%05d.dat" % iii, spectra=spectra)
            diff_list += [Db_a_sim[m] - Db_b_sim[m]]
        mean, std = np.mean(diff_list, axis=0), np.std(diff_list, axis=0)
        
        
        chi2 = np.sum(diff ** 2 / std ** 2)
        nDoF = len(lb)
        
        
        plt.figure(figsize=(16, 8))
        plt.errorbar(lb, diff * lb ** pow[m], std * lb ** pow[m], fmt=".", label=r"$\chi^{2}$ = %.02f/%d" % (chi2, nDoF))
        if overplot_theory_lines[m] is not None:
            l, cl = overplot_theory_lines[m]
            plt.plot(l, cl * l ** pow[m], color="red")
        plt.ylabel(r"$\ell^{%s} D_{\ell}^{%s}$" % (pow[m], m), fontsize=22)
        plt.title(f"{spec_a}  -  {spec_b}", fontsize=22)
        plt.legend(fontsize=22)
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.plot(lb, lb*0, color="gray")
        plt.savefig(f"{spec_a}_{spec_b}_{m}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
