"""
Plots all kinds of spectra of combination A x B in the case where B has no splits
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

# Define spectra path and template to read it
try: 
    d.read_from_file(sys.argv[1])
    spectra_path = sys.argv[2]
    yaml_path = sys.argv[3]
except:
    spectra_path = '/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1022_type1'
    spectra_path = '/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1019_maskglitch'
    spectra_path = '/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1022_all_xmask_type2'
    d.read_from_file(spectra_path + '/_paramfile.dict')
    yaml_path = 'python/plots_1019.yaml'

with open(yaml_path, "r") as f:
    plot_info: dict = yaml.safe_load(f)

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
Dls_auto = {}
for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(surveys_arrays, r=2):
    if (sv_ar1 in surveys_arrays_A):
        ls, Dls_cross[f'{sv_ar1}x{sv_ar2}'] = so_spectra.read_ps(spectra_cross_template.format(sv_ar1, sv_ar2), spectra=spectra)
    elif (sv_ar1 in surveys_arrays_B):
        ls, Dls_auto[f'{sv_ar1}x{sv_ar2}'] = so_spectra.read_ps(spectra_auto_template.format(sv_ar1, sv_ar2), spectra=spectra)

# Define noise as BxB - AxB with same freq (or 150)
Dls_noise = {}
for freq in ['090', '150', '220', '280']:
    for sv_ar in surveys_arrays_B:
        if freq in sv_ar:
            # Take first iteration of sr_ar in surveysA, if not take 150
            try:
                sv_ar_A = [_ for _ in surveys_arrays_A if freq in _][0] 
            except:
                sv_ar_A = [_ for _ in surveys_arrays_A if '150' in _][0] 
            print(f'noise_{sv_ar} = {sv_ar}x{sv_ar} - {sv_ar_A}x{sv_ar}')
            Dls_noise[sv_ar] = {k: Dls_auto[f'{sv_ar}x{sv_ar}'][k] - Dls_cross[f'{sv_ar_A}x{sv_ar}'][k] for k in spectra}

fac = ls * (ls + 1) / (2 * np.pi)

clfile = '/pscratch/sd/m/merrydup/pipe0004_BN/spectra/LCDM_spectra.txt'
l, ps_theory = so_spectra.read_ps(clfile, spectra=spectra)

# AxB cross plots
for f in spectra:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f] * l * (l + 1) / (2 * np.pi), color='black', label='theory')

    for sv_ar_comb, Dls in Dls_cross.items():
        ax.plot(ls, Dls[f], label=sv_ar_comb)
    
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

# BxB auto plots
for f in spectra:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f] * l * (l + 1) / (2 * np.pi), color='black', label='theory')

    for sv_ar1, sv_ar2 in itertools.combinations_with_replacement(surveys_arrays_B, r=2):
            ax.plot(ls, Dls_auto[f'{sv_ar1}x{sv_ar2}'][f], label=f'{sv_ar1}x{sv_ar2}')
    
    ax.set_xlabel(r'$\ell$', fontsize=18)
    ax.set_ylabel(fr'$D^{{{f}}}_\ell$', fontsize=18)
    ax.set_title(f)
    ax.set_yscale(plot_info['yscale'][f])
    ax.set_ylim(*plot_info['cross_AxB']['ylims'][f])
    ax.set_xlim(*plot_info['cross_AxB']['xlims'][f])
    ax.legend()
    
    plt.savefig(save_path_cross + f'auto_{survey_B}x{survey_B}_{f}')
    plt.close()


# BxB noise plots
for f in spectra_auto:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f], color='black', label='theory')

    for sv_ar2 in surveys_arrays_B:
        ax.plot(ls, Dls_noise[f'{sv_ar2}'][f] / fac, label=f'{sv_ar}x{sv_ar} - {sv_ar_A}x{sv_ar}')

    ax.set_xlabel(r'$\ell$', fontsize=18)
    ax.set_ylabel(fr'$N^{{{f}}}_\ell$', fontsize=18)
    ax.set_title(f)
    ax.set_yscale('log')
    ax.set_ylim(*plot_info['noise_BxB']['ylims'][f])
    ax.set_xlim(*plot_info['noise_BxB']['xlims'][f])
    ax.legend(loc=(1.01, 0.))
    plt.savefig(save_path_cross_noises + f'noise_{survey_A}x{survey_B}_{f}')
    plt.close()


# BxB noise plots
Nls = {}
rms_list = [5, 10, 15, 20, 25, 30, 50, 100, 200]
for rms in rms_list:
    ls_nls, Nls[rms] = pspy_utils.get_nlth_dict(rms, type='Cl', lmax=8000, spectra=spectra)

# Choose a colormap and create a list of colors
cmap = plt.get_cmap('viridis')   # try 'plasma', 'coolwarm', 'turbo', etc.
norm = Normalize(vmin=0, vmax=len(rms_list) - 1)
colors = [cmap(norm(i)) for i in range(len(rms_list))]

for f in spectra_auto:
    fig, ax = plt.subplots(dpi=150, figsize=(7, 4))
    ax.plot(l, ps_theory[f], color='black', label='theory')
    for sv_ar2 in surveys_arrays_B:
        ax.plot(ls, Dls_noise[f'{sv_ar2}'][f] / fac * beams[sv_ar2]**2, label=f'{sv_ar}x{sv_ar} - {sv_ar_A}x{sv_ar}')

    for i, (rms, nls) in enumerate(Nls.items()):
        ax.plot(ls_nls, nls[f], label=f'{rms}', color=colors[i], lw=4, alpha=0.4, zorder=-10)

    ax.set_xlabel(r'$\ell$', fontsize=18)
    ax.set_ylabel(fr'$N^{{{f}}}_\ell$', fontsize=18)
    ax.set_title(f)
    ax.set_yscale('log')
    ax.set_ylim(*plot_info['noise_rms_BxB']['ylims'][f])
    ax.set_xlim(*plot_info['noise_rms_BxB']['xlims'][f])
    ax.legend(loc=(1.01, 0.))
    plt.savefig(save_path_cross_noises + f'noise_rms_{survey_A}x{survey_B}_{f}')
    plt.close()