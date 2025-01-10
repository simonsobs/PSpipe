"""
This script performs EB and TB null tests and additionaly fit for a polarisation angle
"""

from pspy import so_dict, pspy_utils, so_cov, so_spectra
from pspipe_utils import  covariance, pspipe_list, pol_angle
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
from null_infos import *
from cobaya.run import run
import getdist.plots as gdplt
from getdist.mcsamples import loadMCSamples

def get_covariance(cov_dir, spec_name, use_mc_cov, use_beam_cov, use_leakage_cov):
    cov = np.load(f"{cov_dir}/analytic_cov_{spec_name}_{spec_name}.npy")
    
    if use_mc_cov == True:
        mc_cov = np.load(f"{cov_dir}/mc_cov_{spec_name}_{spec_name}.npy")
        cov = covariance.correct_analytical_cov(cov,
                                                mc_cov,
                                                only_diag_corrections=True)
    if use_beam_cov == True:
        beam_cov = np.load(f"{cov_dir}/beam_cov_{spec_name}_{spec_name}.npy")
        cov += beam_cov
        
    if use_leakage_cov == True:
        leakage_cov = np.load(f"{cov_dir}/leakage_cov_{spec_name}_{spec_name}.npy")
        cov += leakage_cov
    
    return cov


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

data = True

unblinding_dir = "EB_TB_unblinding_results"
pspy_utils.create_directory(unblinding_dir)


type = d["type"]
spec_list = pspipe_list.get_spec_name_list(d, delimiter="_")

cov_dir = "covariances"
bestfit_dir = "best_fits"
if data == True:
    spec_dir = "spectra_leak_corr"
else:
    spec_dir = d["sim_spec_dir"] 

use_mc_cov = True
use_beam_cov = True
use_leakage_cov = True

spec_order = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
ylim = {}
ylim["EB"] = ylim["BE"] =  [-5, 5]
ylim["TB"] = ylim["BT"] = [-10, 10]

bf_phi = {}
bf_phi["dr6_pa5_f090"] = 0.07
bf_phi["dr6_pa5_f150"] = 0.28
bf_phi["dr6_pa6_f090"] = 0.17
bf_phi["dr6_pa6_f150"] = 0.18



samples = []
arrays = []
for spec_name in spec_list:

    if "pa4" in spec_name:
        continue
        
    if data == True:
        spec_name_cross= f"{type}_{spec_name}_cross"
    else:
        spec_name_cross= f"{type}_{spec_name}_cross_00001"

    lb, ps = so_spectra.read_ps(spec_dir + f"/{spec_name_cross}.dat", spectra=spec_order)
    
    lth, psth = so_spectra.read_ps(bestfit_dir + f"/cmb_and_fg_{spec_name}.dat", spectra=spec_order)
    
    lb, psth_b = so_spectra.bin_spectra(lth,
                                        psth,
                                        d["binning_file"],
                                        d["lmax"],
                                        type="Cl",
                                        spectra=spec_order)


    n_bins = len(lb)
    
    cov = get_covariance(cov_dir, spec_name, use_mc_cov, use_beam_cov, use_leakage_cov)
    
    for spec in ["EB", "BE", "TB", "BT"]:

        ms0, ms1 = spec_name.split("x")
        
        _, psth_rot = pol_angle.rot_theory_spectrum(lb, psth_b, bf_phi[ms0], bf_phi[ms1])

        
        if (ms0 == ms1):
            if (spec == "BE") or (spec == "BT"):
                continue
            
        sub_cov = so_cov.selectblock(cov.copy(), spec_order, n_bins, block=spec+spec)
        
        m0, m1 = spec
        min0, max0 = multipole_range[ms0][m0]
        min1, max1 = multipole_range[ms1][m1]
        lmin = max(min0, min1)

        id = np.where(lb > lmin)[0]
    
        chi2 =  ps[spec][id] @ np.linalg.inv(sub_cov[np.ix_(id, id)]) @ ps[spec][id]
        chi2_bf =  (ps[spec][id] - psth_rot[spec][id]) @ np.linalg.inv(sub_cov[np.ix_(id, id)]) @ (ps[spec][id] - psth_rot[spec][id])
        ndof = len(lb[id])
        pte = 1 - ss.chi2(ndof).cdf(chi2)
        pte_bf = 1 - ss.chi2(ndof-1).cdf(chi2_bf)

        std = np.sqrt(sub_cov.diagonal())
        sub_corr = so_cov.cov2corr(sub_cov, remove_diag=True)
        
        if spec in ["EB", "TB"]:
            phi = bf_phi[ms1]
        else:
            phi = bf_phi[ms0]


        plt.figure(figsize=(22,10))
        plt.subplot(1,2,1)
        
        plt.plot(lb, lb * 0, label=f"[$\chi^2 = {{{chi2:.1f}}}/{{{ndof}}}$ (${{{pte:.4f}}}$)], $\phi=0$", color="gray", linestyle="-")
        plt.plot(lb, psth_rot[spec], label=f"[$\chi^2 = {{{chi2_bf:.1f}}}/{{{ndof-1}}}$ (${{{pte_bf:.4f}}}$), $\phi_B={phi:.3f}$]", color="black", linestyle="--")

        plt.errorbar(lb, ps[spec], std,
                     ls="None", marker = ".", ecolor = "navy",
                     color="darkorange")
    
        plt.axvspan(xmin=0, xmax=lmin,  color="gray", alpha=0.7)
        
        ms0 = ms0.replace("dr6_", "")
        ms1 = ms1.replace("dr6_", "")

        title = f"{m0} {ms0} x {m1} {ms1}"
        plt.ylabel(r"$ D_\ell^\mathrm{%s%s}$" % (m0, m1), fontsize=18)
        plt.xlabel(r"$\ell$", fontsize=18)

        if (spec == "EB") or (spec == "TB"):
            title = f"{m1} {ms1} x {m0} {ms0}"
            plt.ylabel(r"$ D_\ell^\mathrm{%s%s}$" % (m1, m0), fontsize=18)

        plt.title(title, fontsize=18)
        plt.legend(fontsize=12)

        plt.legend(fontsize=18)
        plt.xlim(0,7000)
        plt.ylim(ylim[spec])
        
        fname = f"{m0}{m1}_{ms0}x{ms1}.png"
        if (spec == "EB") or (spec == "TB"):
            fname = f"{m1}{m0}_{ms1}x{ms0}.png"

        plt.subplot(1,2,2)
        plt.imshow(sub_corr)
        plt.colorbar()
        plt.savefig(f"{unblinding_dir}/{fname}", bbox_inches="tight" )
        plt.clf()
        plt.close()
        #sys.exit()
        if (spec=="EB") & (ms0 == ms1):
            arrays += [ms0.replace("dr6_", "_")]
            def loglike(phi):
                _, psth_rot = pol_angle.rot_theory_spectrum(lb, psth_b, phi, phi)
                res = ps[spec][id] - psth_rot[spec][id]
                chi2 =  res @ np.linalg.inv(sub_cov[np.ix_(id, id)]) @ res

                return -0.5 * chi2
    
            info = {}
            info["likelihood"] = { "my_like": loglike}
            info["params"] = { "phi": {  "prior": {  "min": -0.5,  "max": 0.5},  "ref": 0,  "proposal": 0.005, "latex": "\phi"} }
            info["sampler"] = {  "mcmc": {  "max_tries": 1e6,  "Rminus1_stop": 0.001, "Rminus1_cl_stop": 0.01}}
            info["output"] = f"{unblinding_dir}/mcmc_{ms0}"
            info["force"] = True
            info["debug"] = False
            updated_info, sampler = run(info)
            samples += [loadMCSamples(f"{unblinding_dir}/mcmc_{ms0}", settings={"ignore_rows": 0.5})]
        


g = gdplt.get_subplot_plotter(width_inch=12)
g.plot_1d(samples, param := "phi", lims=[-0.3, 1])
g.add_legend(
    [
        f"{wafer}: $\phi$ = {s.mean(param):.2f} $\pm$ {s.std(param):.2f}"
        for wafer, s in zip(arrays, samples)
    ],
    colored_text=True,
)

plt.savefig(f"{unblinding_dir}/posterior_dist.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()
