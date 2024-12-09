"""
plot the different components of each spectra and compare with the pspipe prediction.
"""
import sys

import pylab as plt
import numpy as np
from pspipe_utils import pspipe_list, log, best_fits
from pspy import so_dict, so_spectra, pspy_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spec_list = pspipe_list.get_spec_name_list(d, delimiter="_")

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

#my_run = "no_kspace"
my_run = "std"

plot_dir = f"results_spectra_components_{my_run}"
pspy_utils.create_directory(plot_dir)

cosmo_params = d["cosmo_params"]
fg_norm = d["fg_norm"]
fg_params = d["fg_params"]
fg_components = d["fg_components"]
do_bandpass_integration = d["do_bandpass_integration"]
lmax = d["lmax"]
type = d["type"]
map_set_list = pspipe_list.get_map_set_list(d)

passbands, band_shift_dict = {}, {}
for map_set in map_set_list:
    freq_info = d[f"freq_info_{map_set}"]
    nu_ghz, pb = np.loadtxt(freq_info["passband"]).T
    passbands[f"{map_set}"] = [nu_ghz, pb]
    band_shift_dict[f"bandint_shift_{map_set}"] = d[f"bandpass_shift_{map_set}"]

l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500, **d["accuracy_params"])

fg_dict = best_fits.get_foreground_dict(l_th,
                                        passbands,
                                        fg_components,
                                        fg_params,
                                        fg_norm,
                                        band_shift_dict=band_shift_dict)

from_agora_to_pspipe = {}
from_agora_to_pspipe["dustxdust"] = ["dust"]
from_agora_to_pspipe["kszxksz"] = ["kSZ"]
from_agora_to_pspipe["tszxtsz"] = ["tSZ"]
from_agora_to_pspipe["radioxradio"] = ["radio"]
from_agora_to_pspipe["cibxcib"] = ["cibc", "cibp"]
from_agora_to_pspipe["cmb_seed1xcmb_seed1"] = ["cmb"]

# select which component to plot
plot_select =  ["cibxcib", "cmb_seed1xcmb_seed1", "dustxdust", "kszxksz", "tszxtsz", "radioxradio"]
num_colors = len(plot_select)
cm = plt.get_cmap("jet")

for spectrum in ["TT", "EE"]:
    
    if spectrum == "TT":
        components = ["cmb_seed1", "cib", "dust", "ksz", "rksz", "sync", "tsz", "radio"]
    else:
        components = ["cmb_seed1", "dust", "radio"]

    for my_spec in spec_list:
    
        na, nb = my_spec.split("x")

        plt.figure(figsize=(12,8))
        if spectrum in ["TT", "EE", "BB"]:
            plt.semilogy()
            
        plt.title(f"{my_spec} {spectrum}")

        i_comp = 0
        for comp1 in components:
            for comp2 in components:
            
                agora_name = f"{comp1}x{comp2}"
                if agora_name in plot_select:

                    l, ps = so_spectra.read_ps(f"spectra_components_{my_run}/Dl_{my_spec}_{agora_name}.dat", spectra=spectra)
                    pspipe_name = from_agora_to_pspipe[agora_name]
                    
                    if pspipe_name == ["cmb"]:
                        ps_pspipe = ps_dict[spectrum]
                    else:
                        ps_pspipe = 0
                        for ps_name in pspipe_name:
                            ps_pspipe += fg_dict[spectrum.lower(), ps_name, na, nb]

                    if comp1 != comp2:
                        plt.plot(l, np.abs(ps[spectrum]), label=agora_name, color=cm(i_comp / num_colors))
                        plt.plot(l_th, np.abs(ps_pspipe), linestyle="--", color=cm(i_comp / num_colors))
                    else:
                        plt.plot(l, ps[spectrum], label=agora_name, color=cm(i_comp / num_colors))
                        plt.plot(l_th, ps_pspipe, linestyle="--",  color=cm(i_comp / num_colors))
                    i_comp += 1
        
        if spectrum == "TT":
            plt.ylim(10**-1, 10**4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{spectrum}_{my_spec}.png")
        plt.clf()
        plt.close()
