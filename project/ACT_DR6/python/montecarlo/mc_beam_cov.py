"""
This script generate a montecarlo simulation of beam uncertainties, it then compares the result with
the analytic beam covariance computed by get_beam_covariance.py
"""


from pspy import pspy_utils, so_dict, so_cov, so_spectra
from pspipe_utils import best_fits, log, pspipe_list,  covariance
import numpy as np
import pylab as plt
import sys, os


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
type = d["type"]

n_sims = 1000
multistep_path = d["multistep_path"]
lmax = d["lmax"]
binning_file = d["binning_file"]
bestfit_dir = "best_fits"
cov_dir = "covariances"
plot_dir = "plots/x_ar_cov"

pspy_utils.create_directory(plot_dir)


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes_for_cov = ["TT", "TE", "ET", "EE"]
else:
    modes_for_cov = spectra


log.info(f"construct best fit for all cross array spectra")
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

l_th, _ = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)
min, max = int(np.min(l_th)), int(np.max(l_th)) + 1
print(min, max)

bl, error_modes = {}, {}
for sv in surveys:
    for ar in d[f"arrays_{sv}"]:
        name = f"{sv}_{ar}"
        data = np.loadtxt(d[f"beam_T_{name}"])
        _, bl[name], error_modes[name]  = data[min:max, 0], data[min:max, 1], data[min:max, 2:]

        
psb_sim_all = {}
for iii in range(n_sims):
    log.info(f"Simulation n° {iii:05d}/{n_sims:05d}")
    log.info(f"-------------------------")

    bl_sim = {}
    for sv in surveys:
        for ar in d[f"arrays_{sv}"]:
            name = f"{sv}_{ar}"
            n_modes = error_modes[name].shape[1]
            bl_sim[name] = bl[name] + error_modes[name] @ np.random.randn(n_modes)
            
    for spec_name in spec_name_list:
    
        if iii == 0: psb_sim_all[spec_name] = []

        l_th, ps_th_dict = so_spectra.read_ps(f"{bestfit_dir}/cmb_and_fg_{spec_name}.dat", spectra=spectra)
        name1, name2 = spec_name.split("x")
        
        
        beam_corr = (bl[name1] * bl[name2]) /  (bl_sim[name1] * bl_sim[name2])
       
        psb_sim = {}
        for spec in spectra:
            lb, psb_sim[spec] = pspy_utils.naive_binning(l_th, ps_th_dict[spec] * beam_corr, binning_file, lmax)
                    
        psb_sim_all[spec_name] += [psb_sim]

nbins = len(lb)

for sid1, spec_name1 in enumerate(spec_name_list):
    for sid2, spec_name2 in enumerate(spec_name_list):
        if sid1 > sid2: continue
                
        log.info(f"MC leakage cov {spec_name1} {spec_name2}")
        mean, _, mc_cov = so_cov.mc_cov_from_spectra_list(psb_sim_all[spec_name1], psb_sim_all[spec_name2], spectra=modes_for_cov)
        np.save(f"{cov_dir}/mc_beam_cov_{spec_name1}_{spec_name2}.npy", mc_cov)
        
x_ar_mc_beam_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                cov_dir,
                                                                "mc_beam_cov",
                                                                spectra_order=modes_for_cov,
                                                                remove_doublon=True,
                                                                check_pos_def=False)

np.save(f"{cov_dir}/x_ar_mc_beam_cov.npy", x_ar_mc_beam_cov)

x_ar_beam_cov = np.load(f"{cov_dir}/x_ar_beam_cov.npy")

plt.figure(figsize=(20,12))
plt.subplot(2, 1, 1)
plt.semilogy()
plt.plot(x_ar_mc_beam_cov.diagonal(), ".", label="mc cov")
plt.plot(x_ar_beam_cov.diagonal(), label="analytic cov")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x_ar_mc_beam_cov.diagonal()/x_ar_beam_cov.diagonal(), ".", label="mc cov/analytic cov")
plt.savefig(f"{plot_dir}/beam_cov_diagonal_vs_montecarlo.png", bbox_inches="tight")
plt.clf()
plt.close()

plt.figure(figsize=(20,12))
plt.subplot(1, 2, 1)
plt.imshow(so_cov.cov2corr(x_ar_mc_beam_cov, remove_diag=True))
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(so_cov.cov2corr(x_ar_mc_beam_cov, remove_diag=True))
plt.colorbar()
plt.savefig(f"{plot_dir}/beam_corr_vs_montecarlo.png", bbox_inches="tight")
plt.clf()
plt.close()
