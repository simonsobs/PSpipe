"""
plot the frequency scaling of the different component of the agora and compare with the pspipe fg model

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


plot_dir = f"results_frequency_scaling_{my_run}"
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


l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500)

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

cm = plt.get_cmap("jet")

for spectrum in ["TT", "EE"]:

    if spectrum == "TT":
        components = ["cib", "dust", "ksz", "tsz", "radio"]
    else:
        components = ["dust"]
        
    for comp in components:

        agora_name = f"{comp}x{comp}"
        pspipe_name = from_agora_to_pspipe[agora_name]
        
        my_spec_ref = "dr6_pa6_f150xdr6_pa6_f150"
        na_ref, nb_ref = my_spec_ref.split("x")

        l, ps_ref = so_spectra.read_ps(f"spectra_components_{my_run}/Dl_{my_spec_ref}_{agora_name}.dat", spectra=spectra)
        
        
        ps_pspipe_ref = 0
        for ps_name in pspipe_name:
            ps_pspipe_ref += fg_dict[spectrum.lower(), ps_name, na_ref, nb_ref]

        plt.figure(figsize=(12,8))
        plt.semilogy()
        plt.title(f"{comp} {spectrum} (dashed: PSpipe, solid: Agora)", fontsize=22)
        plt.ylabel(r"$D^{X}_{\ell} \ / \ D^{\rm pa6 \ f150 x pa6 \ f150}_{\ell}$", fontsize=22)
        plt.xlabel(r"$\ell$", fontsize=22)
        
        selected_spec_list = spec_list[::2] # don't plot all of them to improve plot quality

        num_colors = len(selected_spec_list)

        for i_comp, my_spec in enumerate(selected_spec_list):
    
            na, nb = my_spec.split("x")
        
            l, ps = so_spectra.read_ps(f"spectra_components_{my_run}/Dl_{my_spec}_{agora_name}.dat", spectra=spectra)
            ps_pspipe = 0
            for ps_name in pspipe_name:
                ps_pspipe += fg_dict[spectrum.lower(), ps_name, na, nb]

            s_name = my_spec.replace("dr6_","")
            plt.plot(l, ps[spectrum]/ps_ref[spectrum], label=s_name, color=cm(i_comp / num_colors))
            plt.plot(l_th, ps_pspipe / ps_pspipe_ref, color=cm(i_comp / num_colors), linestyle="--")
            
        plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{comp}_{spectrum}.png")
        plt.clf()
        plt.close()

