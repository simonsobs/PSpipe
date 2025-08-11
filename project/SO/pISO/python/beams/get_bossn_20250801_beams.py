from pspy import so_dict

from pixell import curvedsky

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

lmax = 10800
plt_lmax = 8000
l = np.arange(0, lmax+1)

plot = False

freq2gauss = {'f090': 3.0, 'f150': 2.5}
freq2gauss_nom = {'f090': 2.2, 'f150': 1.5}
beam_dir = d['beam_dir']

# first make some basic gaussian beams
for fwhm in [3.0, 2.5, 2.2, 1.5, 0.96]:
    bl = hp.gauss_beam(np.deg2rad(fwhm / 60), lmax)
    np.savetxt(f'{beam_dir}/bl_gaussian_{fwhm}arcmin.txt', np.array([l, bl]).T)

# we use the i1 stack from Cristian, and paste a 1/theta^3 wing such that the
# beam reaches the -50dB level. then we calculate the harmonic transform
for freq in freq2gauss:
    r, br, _ = np.loadtxt(f'{beam_dir}/stack_beam_profile_{freq}_i1.txt').T
    r = np.deg2rad(r)
    fwhm = freq2gauss[freq]
    fwhm_nom = freq2gauss_nom[freq]
    sigma = np.deg2rad(fwhm / 60) / np.sqrt(8.0 * np.log(2.0))
    sigma_nom = np.deg2rad(fwhm_nom / 60) / np.sqrt(8.0 * np.log(2.0))
    gauss = np.exp(-0.5 * (r / sigma)**2)
    gauss_nom = np.exp(-0.5 * (r / sigma_nom)**2)

    # fit y=A/r^3 and add the 1/theta^3 wing such that the beam gets to 1e-5
    A = br[-1] * r[-1]**3
    r_max = (A / 1e-5)**(1/3)
    r_ext = np.linspace(r[-1], r_max, 1000)
    r_full = np.concatenate((r[:-1], r_ext))
    br_full = np.concatenate((br[:-1], A / r_ext**3))
    gauss_full = np.exp(-0.5 * (r_full / sigma)**2)
    gauss_nom_full = np.exp(-0.5 * (r_full / sigma_nom)**2)

    # get beam
    bl = curvedsky.profile2harm(br, r, lmax)
    bl_gauss = curvedsky.profile2harm(gauss, r, lmax)
    bl_gauss_nom = curvedsky.profile2harm(gauss_nom, r, lmax)
    bl_full = curvedsky.profile2harm(br_full, r_full, lmax)
    bl_gauss_full = curvedsky.profile2harm(gauss_full, r_full, lmax)
    bl_gauss_nom_full = curvedsky.profile2harm(gauss_nom_full, r_full, lmax)

    bl /= bl[0]
    bl_gauss /= bl_gauss[0]
    bl_gauss_nom /= bl_gauss_nom[0]
    bl_full /= bl_full[0]
    bl_gauss_full /= bl_gauss_full[0]
    bl_gauss_nom_full /= bl_gauss_nom_full[0]

    np.savetxt(f'{beam_dir}/bl_stack_beam_profile_{freq}_i1_ext_wing.txt', np.array([l, bl_full]).T)

    if plot:
        plt.plot(np.rad2deg(r), br, label='data')
        plt.plot(np.rad2deg(r), gauss, label=f'gauss {fwhm} arcmin')
        plt.plot(np.rad2deg(r), gauss_nom, label=f'gauss {fwhm_nom} arcmin', color='pink')
        plt.plot(np.rad2deg(r_full), br_full, label='data ext.', color='C0', linestyle='--')
        plt.plot(np.rad2deg(r_full), gauss_full, label=f'gauss {fwhm} arcmin ext.', color='C1', linestyle='--')
        plt.plot(np.rad2deg(r_full), gauss_nom_full, label=f'gauss {fwhm_nom} arcmin ext.', color='pink', linestyle='--')
        plt.ylim(1e-6, 1)
        plt.legend()
        plt.xlabel('r [deg]')
        plt.semilogy()
        plt.title(f'Cristian i1 freq {freq} real beam')
        plt.grid()
        plt.show()

        l = np.arange(0, lmax+1)
        mask = l <= plt_lmax
        plt.plot(l[mask], bl[mask], label='data')
        plt.plot(l[mask], bl_gauss[mask], label=f'gauss {fwhm} arcmin')
        plt.plot(l[mask], bl_gauss_nom[mask], label=f'gauss {fwhm_nom} arcmin', color='pink')
        plt.plot(l[mask], bl_full[mask], label='data ext.', color='C0', linestyle='--')
        plt.plot(l[mask], bl_gauss_full[mask], label=f'gauss {fwhm} arcmin ext.', color='C1', linestyle='--')
        plt.plot(l[mask], bl_gauss_nom_full[mask], label=f'gauss {fwhm_nom} arcmin ext.', color='pink', linestyle='--')
        plt.xlabel('Multipole l')
        plt.legend()
        plt.title(f'Cristian i1 freq {freq} harmonic beam')
        plt.grid()
        plt.show()

        l = np.arange(0, lmax+1)
        plt.plot(l[mask], bl[mask] / bl_full[mask], label='data / data ext.', color='C0')
        plt.plot(l[mask], bl_gauss[mask] / bl_full[mask], label=f'gauss {fwhm} arcmin  / data ext.', color='C1')
        plt.plot(l[mask], bl_gauss_nom[mask] / bl_full[mask], label=f'gauss {fwhm_nom} arcmin  / data ext.', color='pink')
        plt.axhline(1, color='k', alpha=0.3)
        plt.plot(l[mask], bl_gauss_full[mask] / bl_full[mask], label=f'gauss {fwhm} arcmin ext. / data ext.', color='C1', linestyle='--')
        plt.plot(l[mask], bl_gauss_nom_full[mask] / bl_full[mask], label=f'gauss {fwhm_nom} arcmin ext. / data ext.', color='pink', linestyle='--')
        plt.xlabel('Multipole l')
        plt.legend()
        plt.title(f'Cristian i1 freq {freq} harmonic beam ratio')
        plt.ylim(0.95, 1.75)
        plt.grid()
        plt.show()