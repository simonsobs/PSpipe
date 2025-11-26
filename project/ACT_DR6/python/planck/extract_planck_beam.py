"""
Read the Planck beam files and save the beam as tables in .dat files.
both the intensity, the polarisation and the leakage beam files
"""

import sys

import numpy as np
import pylab as plt
import healpy as hp
from astropy.io import fits
from pspy import pspy_utils, so_dict
from pspipe_utils import log

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

planck_fits_beam_path = d["planck_fits_beam_path"]

freqs = [100, 143, 217, 353]
lmax = 3030
lmax_for_plot = 2000
releases = ["legacy", "npipe_DR6_AxB"]

for release in releases:

    beam_dir = f"beams/{release}"
    pspy_utils.create_directory(beam_dir)

    for freq in freqs:
        
        if release == "npipe_DR6_AxB":
            s1, s2 = "A", "B"
            if freq != 353:
                Wl = fits.open(f"{planck_fits_beam_path}/QP_dr6_pa6_f150/Wl_npipe6v20_{freq}{s1}x{freq}{s2}.fits")
            else:
                print(f"use full sky beam for {freq} GHz")
                Wl = fits.open(f"{planck_fits_beam_path}/quickpol/Wl_npipe6v20_{freq}{s1}x{freq}{s2}.fits")

        if release == "legacy":
            s1, s2 = "hm1", "hm2"
            if freq != 353:
                Wl = fits.open(f"{planck_fits_beam_path}/BeamWf_HFI_R3.01/Wl_R3.01_plikmask_{freq}{s1}x{freq}{s2}.fits")
            else:
                print(f"use full sky beam for {freq} GHz")
                Wl = fits.open(f"{planck_fits_beam_path}/BeamWf_HFI_R3.01/Wl_R3.01_fullsky_{freq}{s1}x{freq}{s2}.fits")

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
        plt.savefig(f"{beam_dir}/beam_{freq}.png")
        plt.clf()
        plt.close()
        
        np.savetxt(f"{beam_dir}/bl_T_{release}_{freq}{s1}x{freq}{s2}.dat", np.transpose([l, bl_T]))
        np.savetxt(f"{beam_dir}/bl_pol_{release}_{freq}{s1}x{freq}{s2}.dat", np.transpose([l, bl_pol]))

        # extract leakage beam
        Wl_TE_2_TE = Wl[4].data["TE_2_TE"][0, :lmax]
        gamma_TE = Wl[1].data["TT_2_TE"][0, :lmax] / Wl_TE_2_TE
        gamma_ET = Wl[1].data["TT_2_ET"][0, :lmax] / Wl_TE_2_TE

        gamma_TB = Wl[1].data["TT_2_TB"][0, :lmax] / Wl_TE_2_TE
        gamma_BT = Wl[1].data["TT_2_BT"][0, :lmax] / Wl_TE_2_TE

        plt.figure(figsize=(12,8))
        plt.title(f"{freq} GHz x {freq} GHz", fontsize=14)
        plt.errorbar(l[:lmax_for_plot], 100 * gamma_TE[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ E_{\rm %s}$" % (release, s1, s2), color="blue", fmt="-", markevery=50)
        plt.errorbar(l[:lmax_for_plot], 100 * gamma_ET[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ E_{\rm %s}$" % (release, s2, s1), color="navy", fmt="+", markevery=50)
        plt.errorbar(l[:lmax_for_plot], 100 * (gamma_TE[:lmax_for_plot] + gamma_ET[:lmax_for_plot]) / 2, label=r"%s $T \ x \ E$" % (release), color="black", fmt="--", markevery=50)

        plt.errorbar(l[:lmax_for_plot], 100 * gamma_TB[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ B_{\rm %s}$" % (release, s1, s2), color="red", fmt="-.", markevery=50)
        plt.errorbar(l[:lmax_for_plot], 100 * gamma_BT[:lmax_for_plot], label=r"%s $T_{\rm %s} \ x \ B_{\rm %s}$" % (release, s2, s1), color="orange", fmt="*", markevery=50)
        plt.errorbar(l[:lmax_for_plot], 100 * (gamma_TB[:lmax_for_plot] + gamma_BT[:lmax_for_plot]) / 2, label=r"%s $T \ x \ B$" % (release), color="gray", fmt="--", markevery=50)

        plt.ylim(-0.8, 0.8)
        plt.ylabel(r"$ \gamma_{\ell}$", fontsize=14)
        plt.legend()
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.savefig(f"{beam_dir}/beam_leakage_{freq}.png")
        plt.clf()
        plt.close()

        zeros = np.zeros(len(l))
    
        np.savetxt(f"{beam_dir}/gamma_{release}_{freq}{s2}_t2e.dat", np.transpose([l, gamma_TE, zeros, zeros, zeros]))
        np.savetxt(f"{beam_dir}/gamma_{release}_{freq}{s1}_t2e.dat", np.transpose([l, gamma_ET, zeros, zeros, zeros]))

        np.savetxt(f"{beam_dir}/gamma_{release}_{freq}{s2}_t2b.dat", np.transpose([l, gamma_TB, zeros, zeros, zeros]))
        np.savetxt(f"{beam_dir}/gamma_{release}_{freq}{s1}_t2b.dat", np.transpose([l, gamma_BT, zeros, zeros, zeros]))

        gamma_mean_TE = (gamma_TE + gamma_ET) / 2
        gamma_mean_TB = (gamma_TB + gamma_BT) / 2

        np.savetxt(f"{beam_dir}/gamma_mean_{release}_{freq}{s1}{s2}_t2e.dat", np.transpose([l, gamma_mean_TE,  zeros, zeros,  zeros]))
        np.savetxt(f"{beam_dir}/gamma_mean_{release}_{freq}{s1}{s2}_t2b.dat", np.transpose([l, gamma_mean_TB,  zeros, zeros,  zeros]))

# these beams will be used for source-sub, they have high ell max

release = "npipe"
beam_dir = f"beams/{release}"
pspy_utils.create_directory(beam_dir)


pl = hp.pixwin(2048)

for freq in freqs:

    Wl_AA = fits.open(f"{planck_fits_beam_path}/quickpol/Wl_npipe6v20_{freq}Ax{freq}A.fits")
    Wl_BB = fits.open(f"{planck_fits_beam_path}/quickpol/Wl_npipe6v20_{freq}Bx{freq}B.fits")

    bl_T_A = np.sqrt(Wl_AA[1].data["TT_2_TT"][0, :])
    bl_T_B = np.sqrt(Wl_BB[1].data["TT_2_TT"][0, :])
    
    bl_T_coadd = (bl_T_A + bl_T_B) / 2

    bl_T_coadd = bl_T_coadd[:len(pl)]
    bl_T_coadd_pixwin = bl_T_coadd * pl
    
    l = np.arange(len(bl_T_coadd))
    

    np.savetxt(f"{beam_dir}/bl_T_{release}_{freq}_coadd.dat", np.transpose([l, bl_T_coadd]))
    np.savetxt(f"{beam_dir}/bl_T_{release}_{freq}_coadd_pixwin.dat", np.transpose([l, bl_T_coadd_pixwin]))


