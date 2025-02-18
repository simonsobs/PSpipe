"""
This script correct the npipe power spectra from correlated residual measured in the AxB npipe simulations or hm1xhm2 simulation in the case of legacy
The correction has been precomputed using
python get_planck_spectra_correction_from_nlms.py global_dr6v4xnpipe.dict
python mc_spectra_analysis.py global_dr6v4xnpipe.dict
and store in the dict variable mc_corr_dir
"""
import matplotlib
matplotlib.use("Agg")

import sys

import numpy as np
from pspipe_utils import leakage, pspipe_list, log
from pspy import  so_dict, so_spectra, pspy_utils
import pylab as plt
from copy import deepcopy

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)

planck_version = d["planck_version"]
mc_corr_dir = d["planck_mc_correction"]
type = d["type"]
n_sims = d["iStop"] - d["iStart"] + 1

spec_dir = "spectra_leak_corr"
spec_corr_dir = "spectra_leak_corr_planck_bias_corr"
bestfit_dir = "best_fits"
plot_dir = "plots/planck_mc_correction/"

pspy_utils.create_directory(spec_corr_dir)
pspy_utils.create_directory(plot_dir)

sname_corr_list = []
corr, std = {}, {}
for i in range(n_spec):
    sv1, ar1, sv2, ar2 = sv1_list[i], ar1_list[i], sv2_list[i], ar2_list[i]
    sname = f"{sv1}_{ar1}x{sv2}_{ar2}"
    
    lb, ps = so_spectra.read_ps(f"{spec_dir}/{type}_{sname}_cross.dat", spectra=spectra)
    ps_corr = deepcopy(ps)
    
    if (sv1 == "Planck") & (sv2 == "Planck"):
        log.info(f"correcting  spectra {sname}")

        for spec in spectra:
            lb_c, corr[spec, sname], std[spec, sname] = np.loadtxt(f"{mc_corr_dir}/spectra_{spec}_{sname}_cross.dat",
                                                                         unpack=True)
            
            if not np.all(lb == lb_c): raise ValueError("correction binning != data binning")

            ps_corr[spec] -= corr[spec, sname]
            
        sname_corr_list += [sname]
            
    so_spectra.write_ps(f"{spec_corr_dir}/{type}_{sname}_cross.dat", lb, ps_corr, type, spectra=spectra)
                        
lth, ps_th = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

for spec in spectra:
    plt.figure(figsize=(12, 8))
    for sname in sname_corr_list:
        m_set1, m_set2 = sname.split("x")
            
        if ("f100" in m_set1) or ("f100" in m_set2):
            lmax = 1500
        else:
            lmax = 2000
                
        id = np.where(lb < lmax)
        plt.errorbar(lb[id], corr[spec, sname][id], std[spec, sname][id] / np.sqrt(n_sims), label = f"{sname}")
            
    plt.plot(lth, ps_th[spec] * 1 / 100, color="gray", label= "1% CMB")
    plt.legend()
    plt.ylabel(r"$D_{\ell}^{%s}$" % spec, fontsize=16)
    plt.xlabel(r"$\ell$", fontsize=16)
    plt.savefig(f"{plot_dir}/{planck_version}_correction_{spec}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
