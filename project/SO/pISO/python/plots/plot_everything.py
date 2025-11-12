"""
Plots all kinds of spectra of combination A x B
"""
from pspy import so_spectra, pspy_utils, so_cov, so_map, so_window, so_dict
from math import pi
import numpy as np
import healpy as hp
# import pylab as plt
from matplotlib import pyplot as plt
import os
from pspy import pspy_utils
from matplotlib.colors import Normalize
import itertools
import yaml
import sys
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spectra_auto = ["TT", "EE", "BB"]
binning_file = "/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/pspipe/binning/binning_50"

d = so_dict.so_dict()

with open(f'python/plots_1019.yaml', "r") as f:
    plot_info: dict = yaml.safe_load(f)

# Define spectra path and template to read it
try: 
    d.read_from_file(sys.argv[1])
    spectra_path = sys.argv[2]
except:
    spectra_path = '/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1019_carlos_150'
    d.read_from_file(spectra_path + '/_paramfile.dict')

spectra_cross_template = spectra_path + '/Dl_{}x{}_cross.dat'
spectra_auto_template = spectra_path + '/Dl_{}x{}_auto.dat'
spectra_noise_template = spectra_path + '/Dl_{}x{}_noise.dat'

# Define surveys and arrays to plot
survey_A = 'dr6'
arrays_A = d[f'arrays_{survey_A}']
# arrays_A = ['pa5_f090']
surveys_arrays_A = [f'{survey_A}_{ar}' for ar in arrays_A]

survey_B = 'SO'
arrays_B = d[f'arrays_{survey_B}']
# = ['i1_f090', 'i1_f150']
# arrays_B = ['i1_f090', 'i3_f090', 'i4_f090', 'i6_f090']
surveys_arrays_B = [f'{survey_B}_{ar}' for ar in arrays_B]

beams = {sv_ar: pspy_utils.naive_binning(
    np.loadtxt(d[f'beam_T_{sv_ar}']).T[0],
    np.loadtxt(d[f'beam_T_{sv_ar}']).T[1] / (max(np.loadtxt(d[f'beam_T_{sv_ar}']).T[1])),
    d['binning_file'],
    lmax=8000,
)[1] for sv_ar in surveys_arrays_B}

surveys = [survey_A, survey_B]
surveys_arrays = surveys_arrays_A + surveys_arrays_B

# Define where and what to plot

save_path = spectra_path + '/plots/'
os.makedirs(save_path, exist_ok=True)
save_path_cross = save_path + 'cross/'
os.makedirs(save_path_cross, exist_ok=True)
save_path_cross_freqs = save_path + 'cross_freqs/'
os.makedirs(save_path_cross_freqs, exist_ok=True)
save_path_cross_noises = save_path + 'noises/'
os.makedirs(save_path_cross_noises, exist_ok=True)

# Load spectra
Dls_cross = {}
Dls_noise = {}
for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(surveys_arrays, r=2):
    ls, Dls_cross[f'{sv_ar1}x{sv_ar2}'] = so_spectra.read_ps(spectra_cross_template.format(sv_ar1, sv_ar2), spectra=spectra)
    try:
        ls, Dls_noise[f'{sv_ar1}x{sv_ar2}'] = so_spectra.read_ps(spectra_noise_template.format(sv_ar1, sv_ar2), spectra=spectra)
    except:
        pass
fac = ls * (ls + 1) / (2 * np.pi)

clfile = '/pscratch/sd/m/merrydup/pipe0004_BN/spectra/LCDM_spectra.txt'
l, ps_theory = so_spectra.read_ps(clfile, spectra=spectra)

# AxB cross plots
for f in spectra:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f] * l * (l + 1) / (2 * np.pi), color='black', label='theory')

    for sv_ar1 in surveys_arrays_A:
        for sv_ar2 in surveys_arrays_B:
            ax.plot(ls, Dls_cross[f'{sv_ar1}x{sv_ar2}'][f], label=f'{sv_ar1}x{sv_ar2}')
    
    ax.set_xlabel(r'$\ell$', fontsize=18)
    ax.set_ylabel(fr'$D^{{{f}}}_\ell$', fontsize=18)
    ax.set_title(f)
    ax.set_yscale(plot_info['yscale'][f])
    ax.set_ylim(*plot_info['cross_AxB']['ylims'][f])
    ax.set_xlim(*plot_info['cross_AxB']['xlims'][f])
    ax.legend()
    
    plt.savefig(save_path_cross + f'cross_{survey_A}x{survey_B}_{f}')
    plt.close()

# AxB cross plots per frequency
frequencies = ['090', '150', '220', '280']
for freq1, freq2 in itertools.combinations_with_replacement(frequencies, r=2):
    if True in [freq1 in sv_ar for sv_ar in surveys_arrays]:
        if True in [freq2 in sv_ar for sv_ar in surveys_arrays]:
            for f in spectra:
                fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
                ax.plot(l, ps_theory[f] * l * (l + 1) / (2 * np.pi), color='black', label='theory')

                for sv_ar1 in surveys_arrays_A:
                    for sv_ar2 in surveys_arrays_B:
                        if (freq1 in sv_ar1) & (freq2 in sv_ar2):
                            ax.plot(ls, Dls_cross[f'{sv_ar1}x{sv_ar2}'][f], label=f'{sv_ar1}x{sv_ar2}')
                
                ax.set_xlabel(r'$\ell$', fontsize=18)
                ax.set_ylabel(fr'$D^{{{f}}}_\ell$', fontsize=18)
                ax.set_title(f)
                ax.set_yscale(plot_info['yscale'][f])
                ax.set_ylim(*plot_info['cross_AxB']['ylims'][f])
                ax.set_xlim(*plot_info['cross_AxB']['xlims'][f])
                ax.legend()
                
                plt.savefig(save_path_cross_freqs + f'cross_{survey_A}x{survey_B}_{f}_{freq1}x{freq2}')
                plt.close()

# BxB cross plots
for f in spectra:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f] * l * (l + 1) / (2 * np.pi), color='black', label='theory')

    for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(surveys_arrays_B, r=2):
            ax.plot(ls, Dls_cross[f'{sv_ar1}x{sv_ar2}'][f], label=f'{sv_ar1}x{sv_ar2}')
    
    ax.set_xlabel(r'$\ell$', fontsize=18)
    ax.set_ylabel(fr'$D^{{{f}}}_\ell$', fontsize=18)
    ax.set_title(f)
    ax.set_yscale(plot_info['yscale'][f])
    ax.set_ylim(*plot_info['cross_AxB']['ylims'][f])
    ax.set_xlim(*plot_info['cross_AxB']['xlims'][f])
    ax.legend()
    
    plt.savefig(save_path_cross + f'cross_{survey_B}x{survey_B}_{f}')
    plt.close()


# BxB noise plots
for f in spectra_auto:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f], color='black', label='theory')

    for sv_ar2 in surveys_arrays_B:
        ax.plot(ls, Dls_noise[f'{sv_ar2}x{sv_ar2}'][f] / fac, label=f'{sv_ar2}')

    ax.set_xlabel(r'$\ell$', fontsize=18)
    ax.set_ylabel(fr'$N^{{{f}}}_\ell$', fontsize=18)
    ax.set_title(f)
    ax.set_yscale('log')
    ax.set_ylim(*plot_info['noise_BxB']['ylims'][f])
    ax.set_xlim(*plot_info['noise_BxB']['xlims'][f])
    ax.legend()
    plt.savefig(save_path_cross_noises + f'noise_{survey_A}x{survey_B}_{f}')
    plt.close()


# BxB noise plots
Nls = {}
for rms in [5, 10, 15, 20, 25, 30]:
    ls_nls, Nls[rms] = pspy_utils.get_nlth_dict(rms, type='Cl', lmax=8000, spectra=spectra)

# Choose a colormap and create a list of colors
cmap = plt.get_cmap('viridis')   # try 'plasma', 'coolwarm', 'turbo', etc.
norm = Normalize(vmin=0, vmax=5)
colors = [cmap(norm(i)) for i in range(6)]

for f in spectra_auto:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f], color='black', label='theory')
    for sv_ar2 in surveys_arrays_B:
        ax.plot(ls, Dls_noise[f'{sv_ar2}x{sv_ar2}'][f] / fac * beams[sv_ar2]**2, label=f'{sv_ar2}')

    for i, (rms, nls) in enumerate(Nls.items()):
        ax.plot(ls_nls, nls[f], label=f'{rms}', color=colors[i], lw=4, alpha=0.4, zorder=-10)

    ax.set_xlabel(r'$\ell$', fontsize=18)
    ax.set_ylabel(fr'$N^{{{f}}}_\ell$', fontsize=18)
    ax.set_title(f)
    ax.set_yscale('log')
    ax.set_ylim(*plot_info['noise_rms_BxB']['ylims'][f])
    ax.set_xlim(*plot_info['noise_rms_BxB']['xlims'][f])
    ax.legend()
    plt.savefig(save_path_cross_noises + f'noise_rms_{survey_A}x{survey_B}_{f}')
    plt.close()