"""
This script does montecarlo simulation to estimate the covariance of the spectra due to uncertainties in the
leakage beam model.
We generate a bunch of simulations of the leakage beam, apply it to a theory spectra, and correct assuming the average leakage beam.
This should look like what happens to the actual data.
Note that here we use only the leakage beam associated to split0, this is because for ACT the leakage beam is the same for all split, and for Planck
we don't have an estimate of the beam leakage errors
"""
import matplotlib
matplotlib.use("Agg")

import sys

import numpy as np
from pspipe_utils import leakage, pspipe_list, log
from pspy import  so_dict, so_spectra, pspy_utils, so_cov, so_mpi
import pylab as plt

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
leakage_file_dir = d["leakage_file_dir"]
n_sims = 10000

bestfit_dir = "best_fits"
sim_dir = "montecarlo_beam_leakage"
plot_dir = "plots/leakage/"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(sim_dir)

# read the leakage model
gamma, err_m, var = {}, {}, {}

plt.figure(figsize=(12, 8))
for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
    
        name = f"{sv}_{ar}"
        
        log.info(f"reading leakage info {name}")

        gamma[name], err_m[name], var[name] = {}, {}, {}
        l, gamma[name]["TE"], err_m[name]["TE"], gamma[name]["TB"], err_m[name]["TB"] = leakage.read_leakage_model(leakage_file_dir,
                                                                                                                   d[f"leakage_beam_{name}"][0],
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
        

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

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


so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_sims)

for iii in subtasks:
    log.info(f"Simulation n° {iii:05d}/{n_sims:05d}")
    log.info(f"-------------------------")

    gamma_sim = {}
    for name in gamma.keys():
        gamma_sim[name] = {}
        gamma_sim[name]["TE"] = leakage.leakage_beam_sim(gamma[name]["TE"], err_m[name]["TE"])
        gamma_sim[name]["TB"] = leakage.leakage_beam_sim(gamma[name]["TB"], err_m[name]["TB"])

    for spec_name in spec_name_list:

        name1, name2 = spec_name.split("x")
        
        lb, psb_sim = leakage.leakage_correction(l_th,
                                                ps_th_dict[spec_name],
                                                gamma_sim[name1],
                                                var[name1],
                                                lmax,
                                                gamma_beta=gamma_sim[name2],
                                                binning_file=binning_file)
                                                
        psb_sim_corrected = {}
        for spec in spectra:
            psb_sim_corrected[spec] = psb_sim[spec] - residual[spec_name][spec]
        
        spec_name_cross = f"{type}_{name1}x{name2}_cross_{iii:05d}"
        so_spectra.write_ps(sim_dir + f"/{spec_name_cross}.dat", lb, psb_sim_corrected, type, spectra=spectra)
        
        
