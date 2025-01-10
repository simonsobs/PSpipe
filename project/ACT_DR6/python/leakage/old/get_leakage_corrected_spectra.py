"""
This script correct the power spectra from  T->P leakage
the idea is to subtract from each data spectra the expected contribution from the leakage
computed from the planet beam leakage measurement and a best fit model.
Note that this assume we have a best fit model, but realistically, not knowing the best fit
presicely is going to be a second order correction
"""
import matplotlib
matplotlib.use("Agg")

import sys

import numpy as np
from pspipe_utils import leakage, pspipe_list, log
from pspy import  so_dict, so_spectra, pspy_utils
import pylab as plt

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
binning_file = d["binning_file"]
type = d["type"]
leakage_file_dir = d["leakage_file_dir"]

bestfit_dir = "best_fits"
spec_dir = "spectra"
spec_corr_dir = "spectra_corrected"
plot_dir = "plots/leakage/"

pspy_utils.create_directory(spec_corr_dir)
pspy_utils.create_directory(plot_dir)

# read the leakage model
gamma, var = {}, {}

plt.figure(figsize=(12, 8))
for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
        name = f"{sv}_{ar}"
        gamma[name], var[name] = {}, {}
        l, gamma[name]["TE"], err_m_TE, gamma[name]["TB"], err_m_TB = leakage.read_leakage_model(leakage_file_dir,
                                                                                                 d[f"leakage_beam_{name}"][0],
                                                                                                 lmax,
                                                                                                 lmin=2)

        var[name]["TETE"] = leakage.error_modes_to_cov(err_m_TE).diagonal()
        var[name]["TBTB"] = leakage.error_modes_to_cov(err_m_TB).diagonal()
        var[name]["TETB"] = var[name]["TETE"] * 0

        plt.subplot(2,1,1)
        plt.errorbar(l, gamma[name]["TE"], np.sqrt(var[name]["TETE"]), fmt=".", label=name)
        plt.ylabel(r"$\gamma^{TE}_{\ell}$", fontsize=17)
        plt.legend()
        plt.subplot(2,1,2)
        plt.ylabel(r"$\gamma^{TB}_{\ell}$", fontsize=17)
        plt.xlabel(r"$\ell$", fontsize=17)
        plt.errorbar(l, gamma[name]["TB"], np.sqrt(var[name]["TBTB"]), fmt=".", label=name)
        plt.legend()
        
plt.savefig(f"{plot_dir}/beam_leakage.png", bbox_inches="tight")
plt.clf()
plt.close()

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

for spec_name in spec_name_list:

    name1, name2 = spec_name.split("x")

    log.info(f"correcting spectra {spec_name}")
    
    l_th, ps_th = so_spectra.read_ps(f"{bestfit_dir}/cmb_and_fg_{spec_name}.dat", spectra=spectra)
    id = np.where(l_th < lmax)
    l_th = l_th[id]
    for spec in spectra:
        ps_th[spec] = ps_th[spec][id]

    lb, residual = leakage.leakage_correction(l_th,
                                              ps_th,
                                              gamma[name1],
                                              var[name1],
                                              lmax,
                                              return_residual=True,
                                              gamma_beta=gamma[name2],
                                              binning_file=binning_file)

    lb, ps = so_spectra.read_ps(spec_dir + f"/{type}_{spec_name}_cross.dat", spectra=spectra)
    
    ps_corr = {}
    for spec in spectra:
    
        ps_corr[spec] = ps[spec] - residual[spec]

        plt.figure(figsize=(12, 8))
        plt.plot(lb, ps[spec], label="pre correction")
        plt.plot(lb, ps_corr[spec], label="post correction")
        plt.legend(fontsize=12)
        plt.savefig(f"{plot_dir}/{spec_name}_{spec}.png", bbox_inches="tight")
        plt.legend()
        plt.clf()
        plt.close()

    so_spectra.write_ps(spec_corr_dir + f"/{type}_{spec_name}_cross.dat", lb, ps_corr, type, spectra=spectra)
