"""
Read the Planck beam files and save the beam as tables in .dat files.
both the intensity, the polarisation and the leakage beam files
"""
import numpy as np
import pylab as plt
from astropy.io import fits
from pspy import pspy_utils

freqs = [100, 143, 217]
lmax = 3030
lmax_for_plot = 2000
beam_path = "/global/cfs/cdirs/act/data/tlouis/dr6v4/beams/planck_beams/"
release = "npipe_DR6"

pspy_utils.create_directory(release)


for c1, freq1 in enumerate(freqs):
    for c2, freq2 in enumerate(freqs):
        if c1 != c2: continue
        
        if release == "npipe":
            s1, s2 = "A", "B"
            Wl = fits.open(f"{beam_path}/quickpol/Wl_npipe6v20_{freq1}{s1}x{freq2}{s2}.fits")

        if release == "npipe_DR6":
            s1, s2 = "A", "B"
            Wl = fits.open(f"{beam_path}/QP_dr6_pa6_f150/Wl_npipe6v20_{freq1}{s1}x{freq2}{s2}.fits")

        if release == "legacy":
            s1, s2 = "hm1", "hm2"
            Wl = fits.open(f"{beam_path}/BeamWf_HFI_R3.01/Wl_R3.01_plikmask_{freq1}{s1}x{freq2}{s2}.fits")

        l = np.arange(lmax)
        
        Wl_TT_2_TT = Wl[1].data["TT_2_TT"][0, :lmax]
        Wl_EE_2_EE = Wl[2].data["EE_2_EE"][0, :lmax]
            
        # extract beam and polarised beam
        bl_T = np.sqrt(Wl_TT_2_TT)
        bl_pol = np.sqrt(Wl_EE_2_EE)
        
        plt.figure(figsize=(12,8))
        plt.subplot(2, 1, 1)
        plt.plot(l[:lmax_for_plot], bl_T[:lmax_for_plot], label="temperature beam", color="lightblue")
        plt.errorbar(l[:lmax_for_plot], bl_pol[:lmax_for_plot],  fmt="+", markevery=50, label="pol beam", color="red")
        plt.ylabel(r"$ B_{\ell}$", fontsize=14)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(l[:lmax_for_plot], (bl_T[:lmax_for_plot] / bl_pol[:lmax_for_plot]) ** 2)
        plt.ylabel(r"$ (B^{\rm T}_{\ell}/B^{\rm pol}_{\ell})^{2} $", fontsize=14)
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.savefig(f"beam_{freq1}x{freq2}.png")
        plt.clf()
        plt.close()
        
        np.savetxt(f"{release}/bl_T_{release}_{freq1}{s1}x{freq2}{s2}.dat", np.transpose([l, bl_T]))
        np.savetxt(f"{release}/bl_pol_{release}_{freq1}{s1}x{freq2}{s2}.dat", np.transpose([l, bl_pol]))

        # extract leakage beam
        Wl_TE_2_TE = Wl[4].data["TE_2_TE"][0, :lmax]
        gamma_TE = Wl[1].data["TT_2_TE"][0, :lmax] / Wl_TE_2_TE
        gamma_ET = Wl[1].data["TT_2_ET"][0, :lmax] / Wl_TE_2_TE

        gamma_TB = Wl[1].data["TT_2_TB"][0, :lmax] / Wl_TE_2_TE
        gamma_BT = Wl[1].data["TT_2_BT"][0, :lmax] / Wl_TE_2_TE

        plt.figure(figsize=(12,8))
        plt.title(f"{freq1} GHz x {freq2} GHz", fontsize=14)
        plt.errorbar(l[:lmax_for_plot], 100 * gamma_TE[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ E_{\rm %s}$" % (release, s1, s2), color="blue", fmt="-", markevery=50)
        plt.errorbar(l[:lmax_for_plot], 100 * gamma_ET[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ E_{\rm %s}$" % (release, s2, s1), color="navy", fmt="+", markevery=50)
        plt.errorbar(l[:lmax_for_plot], 100 * gamma_TB[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ B_{\rm %s}$" % (release, s1, s2), color="red", fmt="-.", markevery=50)
        plt.errorbar(l[:lmax_for_plot], 100 * gamma_BT[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ B_{\rm %s}$" % (release, s2, s1), color="orange", fmt="*", markevery=50)
        plt.ylim(-0.8, 0.8)
        plt.ylabel(r"$ \gamma_{\ell}$", fontsize=14)
        plt.legend()
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.savefig(f"beam_leakage_{freq1}x{freq2}.png")
        plt.clf()
        plt.close()

        np.savetxt(f"{release}/gamma_{release}_{freq2}{s2}.dat", np.transpose([l, gamma_TE, gamma_TB]))
        np.savetxt(f"{release}/gamma_{release}_{freq1}{s1}.dat", np.transpose([l, gamma_ET, gamma_BT]))
        
        zeros = np.zeros(len(l))
        np.savetxt(f"{release}/error_modes_gamma_{release}_{freq2}{s2}.dat", np.transpose([l, zeros, zeros, zeros, zeros, zeros, zeros]))
        np.savetxt(f"{release}/error_modes_gamma_{release}_{freq1}{s1}.dat", np.transpose([l, zeros, zeros, zeros, zeros, zeros, zeros]))
