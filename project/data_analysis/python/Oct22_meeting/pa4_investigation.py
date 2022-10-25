"""
Some pa4 investigation using high spectra
/global/cscratch1/sd/tlouis/data_analysis_final/dr6_choi_binning_high_ell
"""
from pspy import so_dict, so_spectra, so_cov, pspy_utils
import matplotlib.pyplot as plt
import numpy as np
import sys


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]

spec_dir = "spectra"
cov_dir = "covariances"
arrays = ["pa4_f150", "pa5_f150", "pa6_f150"]

test_name = ["150_150", "150_220", "90_150"]
spec_150_150 = ["dr6_pa4_f150xdr6_pa4_f150", "dr6_pa5_f150xdr6_pa5_f150", "dr6_pa6_f150xdr6_pa6_f150"]
spec_150_220 = ["dr6_pa4_f150xdr6_pa4_f220", "dr6_pa4_f220xdr6_pa5_f150", "dr6_pa4_f220xdr6_pa6_f150"]
spec_90_150 = ["dr6_pa4_f150xdr6_pa5_f090",  "dr6_pa5_f090xdr6_pa5_f150", "dr6_pa6_f090xdr6_pa6_f150"]

ref_spec = ["dr6_pa4_f150xdr6_pa4_f150", "dr6_pa4_f150xdr6_pa4_f220", "dr6_pa4_f150xdr6_pa5_f090"]
Dl_spec = {}

for test, spec_comb, ref in zip(test_name, [spec_150_150, spec_150_220, spec_90_150], ref_spec):
    plt.figure(figsize=(16, 8))
    plt.semilogy()
    for spec in spec_comb:
    
        if "pa4_f150" in spec:
            fmt =".-"
        else:
            fmt="--"
        l, Dl = so_spectra.read_ps(f"{spec_dir}/Dl_{spec}_cross.dat", spectra=spectra)
        cov = np.load(f"{cov_dir}/analytic_cov_{spec}_{spec}.npy")
        sigma = so_cov.get_sigma(cov, modes, len(l), "TT")
        Dl_spec[spec] = Dl["TT"]

        plt.errorbar(l, Dl_spec[spec], sigma, label=spec, fmt=fmt)
    plt.ylabel(r"$D_{\ell}^{TT}$", fontsize=22)
    plt.xlabel(r"$\ell$", fontsize=22)
    plt.ylim(30, 4500)
    plt.legend()
    plt.savefig(f"pa4_{test}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure(figsize=(16, 8))

    for spec in spec_comb:
        plt.errorbar(l, Dl_spec[ref]/Dl_spec[spec], label=f"{ref}/{spec}")
    plt.ylim(0.7, 1.3)
    plt.legend()
    plt.savefig(f"pa4_ratio_{test}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
