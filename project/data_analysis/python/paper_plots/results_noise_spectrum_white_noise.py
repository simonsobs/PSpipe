"""
This script compare the noise spectra with white noise
"""

from pspy import so_dict, pspy_utils, so_spectra, so_cov
from pspipe_utils import covariance, pspipe_list, log, best_fits, external_data, misc
import numpy as np
import pylab as plt
import sys, os
import scipy.stats as ss
from matplotlib import rcParams


rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 20
rcParams["axes.titlesize"] = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

surveys = d["surveys"]
lmax = d["lmax"]
binning_file = d["binning_file"]

noise_dir = "results_noise"

pspy_utils.create_directory(noise_dir)

l_th, ps_th = so_spectra.read_ps("best_fits/cmb.dat", spectra=spectra)

rms_uKarcmin = {}
rms_uKarcmin["dr6_pa4_f220"] = 82
rms_uKarcmin["dr6_pa5_f090"] = 20
rms_uKarcmin["dr6_pa5_f150"] = 23.9
rms_uKarcmin["dr6_pa6_f090"] = 23
rms_uKarcmin["dr6_pa6_f150"] = 28

rms_uKarcmin_combined = 0
for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
        rms_uKarcmin_combined += 1 / rms_uKarcmin[f"{sv}_{ar}"] ** 2
rms_uKarcmin_combined = 1 / np.sqrt(rms_uKarcmin_combined)

print(f"combined noise level: {rms_uKarcmin_combined} uk.arcmin")


for sv in surveys:
    arrays = d[f"arrays_{sv}"]

    for ar in arrays:
        
        map_set = f"{sv}_{ar}"
        spec_name = f"{map_set}x{map_set}"
        
        l, bl = misc.read_beams(d[f"beam_T_{map_set}"], d[f"beam_pol_{map_set}"])
        
        l = l[2 : lmax + 2]
        fac = l * (l + 1) / (2 * np.pi)
        white_Nl = (rms_uKarcmin[f"{map_set}"] * np.pi / (60 * 180)) ** 2 / ( bl["T"][2 : lmax + 2] ** 2) * fac
        lb, white_Nb = pspy_utils.naive_binning(l, white_Nl, binning_file, lmax)
        
        if ar == "combined":
            Nb = {}
            lb, Nb["TT"] = np(f"results_noise/DR6_noise_combined_TT.dat", unpack=True)
            lb, Nb["EE"] = np(f"results_noise/DR6_noise_combined_TT.dat", unpack=True)
        else:
            lb, Nb = so_spectra.read_ps(f"spectra/Dl_{spec_name}_noise.dat", spectra=spectra)

        id = np.where(lb > 300)
            
        plt.figure(figsize=(12,10))
        plt.suptitle(f"{ar.replace('_', ' ')}", fontsize=18)
        plt.subplot(2,1,1)
        plt.semilogy()
        plt.plot(lb[id], Nb["TT"][id], "o", label= f"measured noise TT", color="blue")
        plt.plot(lb[id], Nb["EE"][id], "o", label= f"measured noise EE", color="red")
        plt.plot(lb, white_Nb, color="blue", label=f"white noise fit rms: {rms_uKarcmin[f'{map_set}']} uK.arcmin")
        plt.plot(lb, 2 * white_Nb, color="red", label=r"white noise fit rms: $\sqrt{2}$ x %s uK.arcmin" %  rms_uKarcmin[f"{map_set}"])
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.ylabel(r"$\frac{\ell (\ell + 1)}{2\pi}N_\ell \ [\mu K^{2}]$", fontsize=22)
        plt.ylim(1, 10 ** 5)
        plt.legend(fontsize=16)
        plt.subplot(2,1,2)
        plt.semilogy()
        plt.plot(lb[id], Nb["TT"][id]/white_Nb[id], "o", label= f"ratio measured noise / white noise TT", color="blue")
        plt.plot(lb[id], Nb["EE"][id]/(2 * white_Nb[id]), "o", label= f"ratio measured noise / white noise  EE", color="red")
        plt.plot(lb[id], lb[id] * 0 +1)
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.ylabel(r"$N_\ell \ / \  N^{\rm white}_{\ell}$", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{noise_dir}/DR6_noise_{map_set}.png", dpi=300)
        plt.clf()
        plt.close

