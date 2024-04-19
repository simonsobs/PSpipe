"""
This script correct the power spectra from  T->P leakage
the idea is to subtract from each data spectra the expected contribution from the leakage
computed from the planet beam leakage measurement and a best fit model.
Note that this assume we have a best fit model, but realistically, not knowing the best fit
presicely is going to be a second order correction

This one does it per script, this maybe important for Planck Npipe since split A and split B have different beam leakage
"""
import matplotlib
matplotlib.use("Agg")

import sys

import numpy as np
from pspipe_utils import leakage, pspipe_list, log
from pspy import  so_dict, so_spectra, pspy_utils, so_mpi
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
spec_corr_dir = "spectra_leak_corr"
plot_dir = "plots/leakage/"

lmax_for_plot = 2000

pspy_utils.create_directory(spec_corr_dir)
pspy_utils.create_directory(plot_dir)

# read the leakage model
gamma, var, nsplit = {}, {}, {}
for sv in surveys:

    nsplit[sv] = d[f"n_splits_{sv}"]
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
        name = f"{sv}_{ar}"
        
        plt.figure(figsize=(12, 8))

        for i in range(nsplit[sv]):
            gamma[name, i], var[name, i] = {}, {}
            
            leakage_file = d[f"leakage_beam_{name}"][i]
            log.info(f"Read leakage file for {sv} {ar} {i} : {leakage_file}")

            l, gamma[name, i]["TE"], err_m_TE, gamma[name, i]["TB"], err_m_TB = leakage.read_leakage_model(leakage_file_dir,
                                                                                                           leakage_file,
                                                                                                           lmax,
                                                                                                           lmin=2)
            log.info(gamma[name, i]["TB"])
            
            var[name, i]["TETE"] = leakage.error_modes_to_cov(err_m_TE).diagonal()
            var[name, i]["TBTB"] = leakage.error_modes_to_cov(err_m_TB).diagonal()
            var[name, i]["TETB"] = var[name, i]["TETE"] * 0

            id = np.where(l < lmax_for_plot)
            plt.subplot(2,1,1)
            plt.errorbar(l[id], gamma[name, i]["TE"][id], label=f"{name}_{i}")
            plt.ylabel(r"$\gamma^{TE}_{\ell}$", fontsize=17)
            plt.legend()
            plt.subplot(2,1,2)
            plt.ylabel(r"$\gamma^{TB}_{\ell}$", fontsize=17)
            plt.xlabel(r"$\ell$", fontsize=17)
            plt.errorbar(l[id], gamma[name, i]["TB"][id], label=f"{name}_{i}")
        plt.legend()
        plt.savefig(f"{plot_dir}/beam_leakage_{sv}_{ar}_per_split.png", bbox_inches="tight")
        plt.clf()
        plt.close()
        
        
        
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_spec - 1)

for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]
    
    name1 = f"{sv1}_{ar1}"
    name2 = f"{sv2}_{ar2}"

    ps_dict = {}
    for spec in spectra:
        ps_dict[spec, "auto"] = []
        ps_dict[spec, "cross"] = []

    
    l_th, ps_th = so_spectra.read_ps(f"{bestfit_dir}/cmb_and_fg_{sv1}_{ar1}x{sv2}_{ar2}.dat", spectra=spectra)
    id = np.where(l_th < lmax)
    l_th = l_th[id]
    for spec in spectra:
        ps_th[spec] = ps_th[spec][id]


    for s1 in range(nsplit[sv1]):
        for s2 in range(nsplit[sv2]):
            if (sv1 == sv2) & (ar1 == ar2) & (s1 > s2) : continue
            spec_name=f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_{s1}{s2}"
            l, ps = so_spectra.read_ps(spec_dir + f"/{spec_name}.dat", spectra=spectra)
            
            lb, residual = leakage.leakage_correction(l_th,
                                                      ps_th,
                                                      gamma[name1, s1],
                                                      var[name1, s1],
                                                      lmax,
                                                      return_residual=True,
                                                      gamma_beta=gamma[name2, s2],
                                                      binning_file=binning_file)

            ps_corr = {}
            for spec in spectra:
                ps_corr[spec] = ps[spec] - residual[spec]

                id = np.where(lb < lmax_for_plot)
                plt.figure(figsize=(12, 8))
                plt.plot(lb[id], residual[spec][id], label="correction")
                plt.legend(fontsize=12)
                plt.savefig(f"{plot_dir}/{spec_name}_{spec}.png", bbox_inches="tight")
                plt.legend()
                plt.clf()
                plt.close()


            so_spectra.write_ps(spec_corr_dir + f"/{spec_name}.dat", lb, ps_corr, type, spectra=spectra)

            for count, spec in enumerate(spectra):
                if (s1 == s2) & (sv1 == sv2):
                    if count == 0: log.debug(f"[{task}] auto {sv1}_{ar1} x {sv2}_{ar2} {s1}{s2}")
                    ps_dict[spec, "auto"] += [ps_corr[spec]]
                else:
                    if count == 0: log.debug(f"[{task}] cross {sv1}_{ar1} x {sv2}_{ar2} {s1}{s2}")
                    ps_dict[spec, "cross"] += [ps_corr[spec]]

    ps_dict_auto_mean = {}
    ps_dict_cross_mean = {}
    ps_dict_noise_mean = {}

    for spec in spectra:
        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)
        spec_name_cross = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_cross"

        if ar1 == ar2 and sv1 == sv2:
            # Average TE / ET so that for same array same season TE = ET
            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.

        if sv1 == sv2:
            ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
            spec_name_auto = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_auto"
            ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplit[sv1]
            spec_name_noise = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_noise"

    so_spectra.write_ps(spec_corr_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)
    if sv1 == sv2:
        so_spectra.write_ps(spec_corr_dir + f"/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
        so_spectra.write_ps(spec_corr_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)
