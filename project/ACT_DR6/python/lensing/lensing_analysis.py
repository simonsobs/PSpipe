"""
This analyses the lensing sims and compute their covariances, it then compares them with analytic estimate using Amanda MacInnis code
"""
import numpy as np
import pylab as plt
import camb
from pspy import pspy_utils, so_spectra, so_cov, so_map, so_mcm, so_dict
from pspipe_utils import  get_data_path, log
import sys, time


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

test_array = "pa5_f090"
iStart = d["iStart"]
iStop = d["iStop"]
binning_file = d["binning_file"]
cosmo_params = d["cosmo_params"]
accuracy_pars = d["accuracy_params"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

cov_dir ="covariances"
lensing_dir = "lensing"

plot_dir = "plots/lensing"
pspy_utils.create_directory(plot_dir)


l, ps_lensed = so_spectra.read_ps(f"{lensing_dir}/ps_lensed.dat", spectra=spectra)

run_names = ["gaussian", "non_gaussian"]

mc_cov = {}
mean = {}
for run_name in run_names:

    vec_mean = 0
    cov_mean = 0
    nsim = 0
    for iii in range(iStart, iStop + 1):

        lb, Db = so_spectra.read_ps(f"{lensing_dir}/spectra_{run_name}_{iii:05d}.dat", spectra=spectra)
        vec = []
        for spec in spectra:
            vec = np.append(vec, Db[spec])
        
        vec_mean += vec
        cov_mean += np.outer(vec, vec)
        nsim += 1

    vec_mean /= nsim
    mc_cov[run_name] = cov_mean / nsim - np.outer(vec_mean, vec_mean)
    
    n_bins = len(lb)
    mean[run_name] = {}
    for count, spec in enumerate(spectra):
        mean[run_name][spec] = vec_mean[count * n_bins : (count + 1) * n_bins]

log.info(f"We are analysing {nsim} simulations")


analytic_cov = {}
analytic_cov["gaussian"] = np.load(f"{lensing_dir}/analytic_cov.npy")
analytic_non_gaussian  = np.load(f"{cov_dir}/non_gaussian_lensing_cov_dr6_{test_array}xdr6_{test_array}_dr6_{test_array}xdr6_{test_array}.npy")

analytic_cov["non_gaussian"] = analytic_cov["gaussian"] + analytic_non_gaussian

plt.figure(figsize=(16,10))
plt.subplot(1, 2, 1)
plt.title("Analytic", fontsize=16)
plt.imshow(so_cov.cov2corr(analytic_cov["non_gaussian"]), vmin=-0.2, vmax=0.2)
plt.xticks(ticks=np.arange(9) * n_bins + n_bins/2, labels = spectra, rotation=90, fontsize=20)
plt.yticks(ticks=np.arange(9) * n_bins + n_bins/2, labels = spectra, fontsize=20)
plt.subplot(1, 2, 2)
plt.title("MC", fontsize=16)
plt.imshow(so_cov.cov2corr(mc_cov["non_gaussian"]), vmin=-0.2, vmax=0.2)
plt.xticks(ticks=np.arange(9) * n_bins + n_bins/2, labels = spectra, rotation=90, fontsize=20)
plt.yticks(ticks=np.arange(9) * n_bins + n_bins/2, labels = spectra, fontsize=20)
plt.savefig(f"{plot_dir}/analytic_vs_mc_corr.png")
plt.clf()
plt.close()


for spec in ["TT", "TE", "ET", "EE", "BB"]:
    
    fac = {spec: 1 for spec in spectra}
    fac["TT"] = lb **2
    fac["TE"] = lb
    fac["EE"] = lb
    fac["BB"] = lb

    plt.figure(figsize=(12,8))
    for run_name in run_names:
        sub_mc_cov = so_cov.selectblock(mc_cov[run_name], spectra, n_bins, block=spec+spec)
        plt.errorbar(lb, mean[run_name][spec] * fac[spec], np.sqrt(sub_mc_cov.diagonal()), fmt=".")
    plt.plot(l, ps_lensed[spec])
    plt.savefig(f"{plot_dir}/spectrum_{spec}.png")
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(16,12))
    plt.suptitle(spec)
    diag, diag_analytic = [], []

    for run_name in run_names:
        sub_mc_cov = so_cov.selectblock(mc_cov[run_name], spectra, n_bins, block=spec+spec)
        sub_analytic_cov = so_cov.selectblock(analytic_cov[run_name], spectra, n_bins, block=spec+spec)

        diag += [sub_mc_cov.diagonal()]
        diag_analytic += [sub_analytic_cov.diagonal()]

    plt.subplot(1, 3, 1)
    
    plt.plot(lb, diag[1] / diag[0], label="cov diag non gaussian/gaussian (MC)")
    plt.plot(lb, diag_analytic[1] / diag_analytic[0], label="cov diag non gaussian/gaussian (analytic)")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.title("MC corr")
    sub_mc_cov = so_cov.selectblock(mc_cov["non_gaussian"], spectra, n_bins, block=spec+spec)
    plt.imshow(so_cov.cov2corr(sub_mc_cov), vmin=-0.08, vmax=0.08, origin="lower")
    plt.xticks(ticks=np.arange(len(lb))[::3], labels = lb[::3], rotation=90)
    plt.yticks(ticks=np.arange(len(lb))[::3], labels = lb[::3])
    plt.colorbar(shrink=0.75)

    plt.subplot(1, 3, 3)
    plt.title("analytic corr")
    sub_analytic_cov = so_cov.selectblock(analytic_cov["non_gaussian"], spectra, n_bins, block=spec+spec)
    plt.imshow(so_cov.cov2corr(sub_analytic_cov), vmin=-0.08, vmax=0.08, origin="lower")
    plt.xticks(ticks=np.arange(len(lb))[::3], labels = lb[::3], rotation=90)
    plt.yticks(ticks=np.arange(len(lb))[::3], labels = lb[::3])
    plt.colorbar(shrink=0.75)
    
    plt.savefig(f"{plot_dir}/gaussian_vs_non_gaussian_{spec}.png")
    plt.clf()
    plt.close()
