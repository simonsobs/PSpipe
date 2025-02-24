"""
This script illustrate the effect of alpha_tSZ on the tSZ power spectrum and compare it to Battaglia
and Agora based on two different bahamas sims
"""
import matplotlib
from scipy import interpolate
from matplotlib import rcParams
import matplotlib as mpl

import sys, os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pspipe_utils import best_fits, log, pspipe_list, external_data
import pspipe_utils
from pspy import pspy_utils, so_dict, so_spectra


labelsize = 14
fontsize = 20


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


tag = d["best_fit_tag"]

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

# first let's get a list of all frequency we plan to study
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# let's create the directories to write best fit to disk and for plotting purpose

fg_norm = d["fg_norm"]
fg_components = d["fg_components"]

passbands, band_shift_dict = {}, {}
do_bandpass_integration = d["do_bandpass_integration"]

if do_bandpass_integration:
    log.info("Doing bandpass integration")

map_set_list = pspipe_list.get_map_set_list(d)

for map_set in map_set_list:
    freq_info = d[f"freq_info_{map_set}"]
    if do_bandpass_integration:
        nu_ghz, pb = np.loadtxt(freq_info["passband"]).T
    else:
        nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.])

    passbands[f"{map_set}"] = [nu_ghz, pb]
    band_shift_dict[f"bandint_shift_{map_set}"] = d[f"bandpass_shift_{map_set}"]
    log.info(f"bandpass shift: {map_set} {band_shift_dict[f'bandint_shift_{map_set}']}")


log.info("Getting foregrounds contribution")

fg_params = d["fg_params"]

lmin = 100
l_th = np.arange(lmin,6000)


ref_array = "dr6_pa5_f150" # can be any not important for this script

#l, tSZ_agora = external_data.get_agora_spectrum(f"{ref_array}x{ref_array}", "tsz", "tsz", spectrum="TT")
#f = interpolate.interp1d(l, tSZ_agora)
#tSZ_agora_interp = f(l_th)

#bahamas_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/bahamas")

l_agora, tSZ_agora_78 = external_data.get_bahamas_tSZ(AGN_heating="7,8")
l_agora, tSZ_agora_80 = external_data.get_bahamas_tSZ(AGN_heating="8,0")
id_agora = np.where(l_agora == 3000)


#### Battaglia tSZ is the case with alpha_tSZ = 0
new_fg_params = deepcopy(fg_params)
new_fg_params["alpha_tSZ"] = 0
fg_dict = best_fits.get_foreground_dict(l_th,
                                        passbands,
                                        fg_components,
                                        new_fg_params,
                                        fg_norm,
                                        band_shift_dict=band_shift_dict)
tsZ_battaglia = fg_dict["tt", "tSZ", ref_array, ref_array]

#### DR6 Best fit alpha_tSZ = -0.6
new_fg_params = deepcopy(fg_params)
new_fg_params["alpha_tSZ"] = -0.6
fg_dict = best_fits.get_foreground_dict(l_th,
                                        passbands,
                                        fg_components,
                                        new_fg_params,
                                        fg_norm,
                                        band_shift_dict=band_shift_dict)
tsZ_dr6 = fg_dict["tt", "tSZ", ref_array, ref_array]


n_lines = 100
l_norm = 3000
alphas = np.linspace(-0.8,-0.4, n_lines)

fig, (ax, cbar_ax) = plt.subplots(ncols=2, figsize=(6, 3.5), gridspec_kw={"width_ratios": [20, 1]}, dpi=100)
cmap = plt.cm.cividis
norm = plt.Normalize(vmin=np.min(alphas), vmax=np.max(alphas))
cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="vertical", ticks=[alphas[0], alphas[-1]])
cbar_ax.text(1.3, -.61, r"$\alpha_{\rm tSZ}$", fontsize=fontsize)
cb1.ax.tick_params(labelsize=labelsize)

for i, alpha in enumerate(alphas):

    new_fg_params = deepcopy(fg_params)
    new_fg_params["alpha_tSZ"] = alpha

    fg_dict = best_fits.get_foreground_dict(l_th,
                                            passbands,
                                            fg_components,
                                            new_fg_params,
                                            fg_norm,
                                            band_shift_dict=band_shift_dict)
                                            
    tsZ_DR6 = fg_dict["tt", "tSZ", "dr6_pa5_f150", "dr6_pa5_f150"]
    ax.plot(l_th, tsZ_DR6 / tsZ_DR6[l_norm - lmin], color=cmap(norm(alpha)), alpha=0.3, linewidth=1) #we normalise all template at l=3000
    
ax.set_ylabel(r"$D^{\rm tSZ}_\ell/D^{\rm tSZ}_{3000}$", fontsize=fontsize)
ax.set_xlabel(r"$\ell$", fontsize=fontsize)
ax.errorbar(l_th, tsZ_dr6/tsZ_dr6[l_norm - lmin], color="black", label=r"DR6 best fit ($\alpha_{tSZ}=-0.6$)", linestyle="--", linewidth=1)
ax.errorbar(l_th, tsZ_battaglia/tsZ_battaglia[l_norm - lmin],  label=r"Battaglia (2012) ($\alpha_{tSZ}=0$)", linestyle='dotted', color="darkorange", linewidth=1)
ax.errorbar(l_agora, tSZ_agora_78/tSZ_agora_78[id_agora],  label=r"Agora (BAHAMAS $T^{\rm heating}_{\rm AGN} = 10^{7.8} $ K)", linestyle='dotted', color="forestgreen", linewidth=1)
ax.errorbar(l_agora, tSZ_agora_80/tSZ_agora_80[id_agora],  label=r"Agora (BAHAMAS $T^{\rm heating}_{\rm AGN} = 10^{8.0} $ K)", linestyle='dotted', color="blue", linewidth=1)
#ax.errorbar(l_th, tSZ_agora_interp/tSZ_agora_interp[l_norm - lmin],  label="Agora (2022)", linestyle="--", color="yellow", linewidth=3)
ax.set_xticks(range(1000, 6000, 1000))
ax.tick_params(labelsize=labelsize)
ax.set_xlim(100, 6000)

plt.gcf().legend(fontsize=labelsize, loc='upper center', bbox_to_anchor=(0.5, 0.02))
plt.tight_layout()
plt.savefig(f"{paper_plot_dir}/tSZ_shape{tag}.pdf", bbox_inches="tight")
plt.clf()
plt.close()
