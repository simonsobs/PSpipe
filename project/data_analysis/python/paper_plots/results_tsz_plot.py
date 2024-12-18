"""
This script illustrate the effect of alpha_tSZ on the tSZ power spectrum and compare it to Battaglia and Agora
"""
import matplotlib
from scipy import interpolate
from matplotlib import rcParams
import matplotlib as mpl

import sys
from copy import deepcopy
import numpy as np
import pylab as plt
from pspipe_utils import best_fits, log, pspipe_list, external_data
from pspy import pspy_utils, so_dict, so_spectra


rcParams["font.family"] = "serif"
rcParams["font.size"] = "18"
rcParams["xtick.labelsize"] = 25
rcParams["ytick.labelsize"] = 25
rcParams["axes.labelsize"] = 25
rcParams["axes.titlesize"] = 25


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


result_dir = "plots/tSZ_shape"
pspy_utils.create_directory(result_dir)



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

l, tSZ_agora = external_data.get_agora_spectrum(f"{ref_array}x{ref_array}", "tSZ", "tSZ", spectrum="TT")
f = interpolate.interp1d(l, tSZ_agora)
tSZ_agora_interp = f(l_th)


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

fig, (ax, cbar_ax) = plt.subplots(ncols=2, figsize=(14, 8), gridspec_kw={"width_ratios": [20, 1]})
cmap = plt.cm.cividis
norm = plt.Normalize(vmin=np.min(alphas), vmax=np.max(alphas))
cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="vertical")
cb1.set_label(r"$\alpha_{\rm tSZ}$",fontsize=35)

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
    ax.plot(l_th, tsZ_DR6 / tsZ_DR6[l_norm - lmin], color=cmap(norm(alpha)), alpha=0.3) #we normalise all template at l=3000
    
ax.set_ylabel(r"$D^{\rm tSZ}_\ell/D^{\rm tSZ}_{3000}$", fontsize=35)
ax.set_xlabel(r"$\ell$", fontsize=35)
ax.errorbar(l_th, tsZ_dr6/tsZ_dr6[l_norm - lmin], color="gray", label=r"DR6 best fit ($\alpha_{tSZ}=-0.6$)", linestyle="-", linewidth=3)
ax.errorbar(l_th, tsZ_battaglia/tsZ_battaglia[l_norm - lmin],  label=r"Battaglia (2012) ($\alpha_{tSZ}=0$)", linestyle="--", color="darkorange", linewidth=3)
ax.errorbar(l_th, tSZ_agora_interp/tSZ_agora_interp[l_norm - lmin],  label="Agora (2022)", linestyle="--", color="darkgreen", linewidth=3)

ax.legend(fontsize=28, loc="lower right")
plt.tight_layout()
plt.savefig(f"{result_dir}/tSZ_shape.png")
#plt.show()
plt.clf()
plt.close()
