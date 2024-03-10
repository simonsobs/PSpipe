"""
This script performs null tests
and plot residual power spectra and a summary PTE distribution
"""

from pspy import so_dict, pspy_utils, so_cov, so_spectra
from pspipe_utils import  pspipe_list, pol_angle, covariance
import matplotlib.pyplot as plt
import numpy as np
import sys


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

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
type = d["type"]

spec_list = pspipe_list.get_spec_name_list(d, delimiter="_")

cov_dir = "covariances"
bestfit_dir = "best_fits"
spec_dir = "spectra_leak_corr"
spec_dir_corr = spec_dir + "_polangle_corr"
plot_dir = "plots/angle_corr"

pspy_utils.create_directory(spec_dir_corr)
pspy_utils.create_directory(plot_dir)

bf_phi = {}
bf_phi["dr6_pa4_f220"] = 0 # we don't use pa4_f220 in pol, so I fix it to zero here
bf_phi["dr6_pa5_f090"] = 0.07
bf_phi["dr6_pa5_f150"] = 0.28
bf_phi["dr6_pa6_f090"] = 0.17
bf_phi["dr6_pa6_f150"] = 0.18


for spec_name in spec_list:

    lb, ps = so_spectra.read_ps(spec_dir + f"/{type}_{spec_name}_cross.dat", spectra=spectra)
    
    lth, psth = so_spectra.read_ps(bestfit_dir + f"/cmb.dat", spectra=spectra)
    
    lb, psth_b = so_spectra.bin_spectra(lth,
                                        psth,
                                        d["binning_file"],
                                        d["lmax"],
                                        type="Cl",
                                        spectra=spectra)



    cov = get_covariance(cov_dir, spec_name, use_mc_cov=True, use_beam_cov=True, use_leakage_cov=True)

    ms0, ms1 = spec_name.split("x")
        
    _, psth_b_rot = pol_angle.rot_theory_spectrum(lb, psth_b, bf_phi[ms0], bf_phi[ms1])
    
    if "pa4" not in spec_name:

        for spec in spectra:
        
            sigma = so_cov.get_sigma(cov.copy(), spectra, len(lb), spectrum=spec)


            res = psth_b_rot[spec] - psth_b[spec]

            plt.figure(figsize=(16,10))
            plt.suptitle(f"{ms0} (phi={bf_phi[ms0]}) x  {ms1} (phi={bf_phi[ms1]})")
            plt.subplot(2,1,1)
            plt.xlim(0,4000)
            if spec == "TT":
                plt.semilogy()
            plt.errorbar(lb - 7, ps[spec], sigma, fmt=".", color="darkorange", label="pre angle correction")
            ps[spec] -= res
            plt.errorbar(lb + 7, ps[spec], sigma, fmt=".", color="red", label="post angle correction")
            plt.legend()
            plt.xlabel(r"$\ell$", fontsize=18)
            plt.ylabel(r"$D_\ell^\mathrm{%s}$" % (spec), fontsize=18)
            if spec in ["EB", "BE", "BB"]:
                plt.ylim(-1.5, 1.5)
                plt.plot(lb, lb*0, color="gray", alpha=0.7)

            plt.subplot(2,1,2)
            plt.xlim(0,4000)

            plt.plot(lb, res, label="correction")
            plt.xlabel(r"$\ell$", fontsize=18)
            plt.ylabel(r"$\Delta D_\ell^\mathrm{%s}$" % (spec), fontsize=18)
            plt.legend()
            plt.savefig(f"{plot_dir}/{spec}_{spec_name}.png", bbox_inches="tight")
            plt.clf()
            plt.close()
    so_spectra.write_ps(spec_dir_corr + f"/{type}_{spec_name}_cross.dat", lb, ps, type, spectra=spectra)
