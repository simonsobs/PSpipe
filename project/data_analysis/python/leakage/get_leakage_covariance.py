"""
This script uses montecarlo simulation to estimate the covariance of the spectra due to uncertainties in the
leakage beam model.
"""
import matplotlib
matplotlib.use("Agg")

import sys

import numpy as np
from pspipe_utils import pspipe_list, log, covariance
from pspy import  so_dict, so_spectra, pspy_utils, so_cov
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
sim_dir =  "montecarlo_beam_leakage"
cov_dir =  "covariances"
plot_dir = "plots/leakage/"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(sim_dir)
pspy_utils.create_directory(cov_dir)

if d["cov_T_E_only"] == True:
    modes_for_cov = ["TT", "TE", "ET", "EE"]
else:
    modes_for_cov = spectra

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

ps_all_corrected = {}
for iii in range(n_sims+1):
    log.info(f"read simulation n° {iii:05d}/{n_sims:05d}")
    log.info(f"-------------------------")

    for spec_name in spec_name_list:
    
        if iii == 0: ps_all_corrected[spec_name] = []

        name1, name2 = spec_name.split("x")
        spec_name_cross = f"{type}_{name1}x{name2}_cross_{iii:05d}"
        lb, psb_sim_corrected = so_spectra.read_ps(sim_dir + f"/{spec_name_cross}.dat", spectra=spectra)
        
        ps_all_corrected[spec_name] += [psb_sim_corrected]

nbins = len(lb)

for sid1, spec_name1 in enumerate(spec_name_list):
    for sid2, spec_name2 in enumerate(spec_name_list):
        if sid1 > sid2: continue
                
        log.info(f"MC leakage cov {spec_name1} {spec_name2}")

        mean, _, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all_corrected[spec_name1], ps_all_corrected[spec_name2], spectra=modes_for_cov)
        
        leakage_cov = np.zeros(mc_cov.shape)
        leakage_cov[nbins:, nbins:] = mc_cov[nbins:, nbins:]
        np.save(f"{cov_dir}/leakage_cov_{spec_name1}_{spec_name2}.npy", leakage_cov)
        
        if sid1 == sid2:
            l_th, ps_th_dict = so_spectra.read_ps(f"{bestfit_dir}/cmb_and_fg_{spec_name1}.dat", spectra=spectra)

            for my_id, spec in enumerate(modes_for_cov):
                sub_cov =  leakage_cov[my_id * nbins: (my_id + 1) * nbins, my_id * nbins: (my_id + 1) * nbins]
                std = np.sqrt(sub_cov.diagonal())
                plt.figure(figsize=(12,8))
                plt.title(f"{spec_name1}x{spec_name2} {spec}")
                plt.errorbar(lb, mean[spec], std, fmt=".", label="corrected mean")
                plt.plot(l_th, ps_th_dict[spec], color="gray")
                plt.legend()
                plt.savefig(f"{plot_dir}/mc_mean_{spec_name1}_{spec}.png", bbox_inches="tight")
                plt.clf()
                plt.close()

            #mc_cov = mc_cov[nbins:, nbins:] #remove TT
            mc_corr = so_cov.cov2corr(leakage_cov)
            plt.figure(figsize=(12,8))
            plt.title(f"cov({spec_name1}, {spec_name2})")
            plt.imshow(mc_corr)
            plt.savefig(f"{plot_dir}/mc_corr_{spec_name1}_{spec_name2}.png", bbox_inches="tight")
            plt.clf()
            plt.close()

log.info(f"construct x arrays  MC leakage cov")

x_ar_leakage_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                cov_dir,
                                                                "leakage_cov",
                                                                spectra_order=modes_for_cov,
                                                                remove_doublon=True,
                                                                check_pos_def=False)

np.save(f"{cov_dir}/x_ar_leakage_cov.npy", x_ar_leakage_cov)
x_ar_leakage_corr = so_cov.cov2corr(x_ar_leakage_cov, remove_diag=True)
so_cov.plot_cov_matrix(x_ar_leakage_corr, file_name=f"{plot_dir}/xar_leakage_corr")
