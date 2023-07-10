"""
This script compute the extra bit of covariance due to uncertainties from beam leakage
We do it using montercarlo simulation, we generate a bunch of simulation of the leakage beam,
apply it to a theory spectra, and correct assuming the average leakage beam.
This should look like what happens to the actual data.
"""

import sys
import time

import numpy as np
from pspipe_utils import leakage, pspipe_list, log
from pspy import  so_dict, so_spectra, pspy_utils, so_cov
import pylab as plt

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
surveys = d["surveys"]
lmax = d["lmax"]
binning_file = d["binning_file"]
leakage_file_dir = d["leakage_file_dir"]
n_sims = 1000

bestfit_dir = "best_fits"
cov_dir = "covariances"
plot_dir = "plots/leakage/"

if d["cov_T_E_only"] == True:
    modes_for_cov = ["TT", "TE", "ET", "EE"]
else:
    modes_for_cov = spectra

pspy_utils.create_directory(plot_dir)

# read the leakage model
gamma, err_m, var = {}, {}, {}
name_list = []

plt.figure(figsize=(12, 8))
for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
    
        name = f"{sv}_{ar}"
        
        log.info(f"reading leakage info {name}")

        gamma[name], err_m[name], var[name] = {}, {}, {}
        l, gamma[name]["TE"], err_m[name]["TE"], gamma[name]["TB"], err_m[name]["TB"] = leakage.read_leakage_model(leakage_file_dir,
                                                                                                                   d[f"leakage_beam_{name}"],
                                                                                                                   lmax,
                                                                                                                   lmin=2)

        cov = {}
        cov["TETE"] = leakage.error_modes_to_cov(err_m[name]["TE"])
        cov["TBTB"] = leakage.error_modes_to_cov(err_m[name]["TB"])

        var[name]["TETE"] = cov["TETE"].diagonal()
        var[name]["TBTB"] = cov["TBTB"].diagonal()
        var[name]["TETB"] = var[name]["TETE"] * 0
        
        for field in ["TE", "TB"]:
            corr = so_cov.cov2corr(cov[field + field])

            plt.figure(figsize=(12, 8))
            plt.imshow(corr)
            plt.title(f"gamma {field} correlation {name}", fontsize=12)
            plt.colorbar()
            plt.savefig(f"{plot_dir}/gamma_{field}_correlation_{name}.png", bbox_inches="tight")
            plt.clf()
            plt.close()
        
        name_list += [name]

spec_name_list = pspipe_list.get_spec_name_list(d, char="_")

ps_th_dict, residual = {}, {}
for spec_name in spec_name_list:

    name1, name2 = spec_name.split("x")
    
    l_th, ps_th_dict[spec_name] = so_spectra.read_ps(f"{bestfit_dir}/cmb_and_fg_{spec_name}.dat", spectra=spectra)
    id = np.where(l_th < lmax)
    
    l_th = l_th[id]
    for spec in spectra:
        ps_th_dict[spec_name][spec] = ps_th_dict[spec_name][spec][id]

    l, residual[spec_name] = leakage.leakage_correction(l_th,
                                                        ps_th_dict[spec_name],
                                                        gamma[name1],
                                                        var[name1],
                                                        lmax,
                                                        return_residual=True,
                                                        gamma_beta=gamma[name2],
                                                        binning_file=binning_file)


ps_all_corrected = {}

for iii in range(n_sims):

    log.info(f"sims {iii}/{n_sims}")

    gamma_sim = {}
    for name in name_list:
        gamma_sim[name] = {}
        gamma_sim[name]["TE"] = leakage.leakage_beam_sim(gamma[name]["TE"], err_m[name]["TE"])
        gamma_sim[name]["TB"] = leakage.leakage_beam_sim(gamma[name]["TB"], err_m[name]["TB"])

    for spec_name in spec_name_list:

        name1, name2 = spec_name.split("x")
        
        if iii == 0: ps_all_corrected[spec_name] = []

        l, psb_sim = leakage.leakage_correction(l_th,
                                                ps_th_dict[spec_name],
                                                gamma_sim[name1],
                                                var[name1],
                                                lmax,
                                                gamma_beta=gamma_sim[name2],
                                                binning_file=binning_file)
                                         

        psb_sim_corrected = {}
        for spec in modes_for_cov:
            psb_sim_corrected[spec] = psb_sim[spec] - residual[spec_name][spec]
        
        ps_all_corrected[spec_name] += [psb_sim_corrected]

for sid1, spec_name1 in enumerate(spec_name_list):
    for sid2, spec_name2 in enumerate(spec_name_list):
        if sid1 > sid2: continue
    
        mean, _, leakage_cov = so_cov.mc_cov_from_spectra_list(ps_all_corrected[spec_name1], ps_all_corrected[spec_name2], spectra=modes_for_cov)
        
        np.save(f"{cov_dir}/leakage_cov_{spec_name1}_{spec_name2}.npy", leakage_cov)
