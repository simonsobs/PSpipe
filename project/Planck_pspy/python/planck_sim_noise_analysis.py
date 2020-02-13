'''
This script analyze the result of the spectra simulation.
'''

import numpy as np
import pylab as plt
from pspy import so_dict, so_spectra, pspy_utils
from itertools import combinations_with_replacement as cwr
import sys
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
iStart = d["iStart"]
iStop = d["iStop"]
type = d["type"]
freqs = d["freqs"]
lmax = d["lmax"]
splits = ["hm1", "hm2"]
exp = "Planck"

ps_model_dir = "noise_model"

if d["use_ffp10"] == True:
    sim_spectra_dir = "sim_spectra_ffp10"
    spec_name_noise_mean = "mean_noise_simffp10"

else:
    sim_spectra_dir = "sim_spectra"
    spec_name_noise_mean = "mean_noise_sim"

pspy_utils.create_directory(ps_model_dir)

bl = {}
for freq in freqs:
    for hm in splits:
        lbeam, bl_T = np.loadtxt(d["beam_%s_%s_T" % (freq, hm)], unpack=True)
        bl[freq, hm, "TT"] = bl_T[:lmax-2]
        lbeam, bl_pol = np.loadtxt(d["beam_%s_%s_pol" % (freq, hm)], unpack=True)
        bl[freq, hm, "EE"] = bl_pol[:lmax-2]
        bl[freq, hm, "BB"] = bl[freq, hm, "EE"]


freq_pairs = []
for cross in cwr(freqs, 2):
    freq_pairs += [[cross[0], cross[1]]]


for fpair in freq_pairs:
    
    f0, f1 = fpair
    name = "%s_%sx%s_%s" % (exp, f0, exp, f1)

    spec_name_cross = "Planck_%sxPlanck_%s-hm1xhm2" % (f0,f1)
    spec_name_auto1 = "Planck_%sxPlanck_%s-hm1xhm1" % (f0,f1)
    spec_name_auto2 = "Planck_%sxPlanck_%s-hm2xhm2" % (f0,f1)

    nl_sim_mean = {s: [] for s in spectra}

    for iii in range(iStart, iStop):
        l, ps_dict_cross = so_spectra.read_ps("%s/sim_spectra_unbin_%s_%04d.dat" % (sim_spectra_dir, spec_name_cross, iii), spectra=spectra)
        l, ps_dict_auto1 = so_spectra.read_ps("%s/sim_spectra_unbin_%s_%04d.dat" % (sim_spectra_dir, spec_name_auto1, iii), spectra=spectra)
        l, ps_dict_auto2 = so_spectra.read_ps("%s/sim_spectra_unbin_%s_%04d.dat" % (sim_spectra_dir, spec_name_auto2, iii), spectra=spectra)
        
        for spec in spectra:
            if (spec == "TT" or spec == "EE" or spec == "BB") & (f0 == f1):
                bl_hm1 = bl[f0, "hm1", spec]
                bl_hm2 = bl[f0, "hm2", spec]
                nth_hm1 = ps_dict_auto1[spec] * bl_hm1**2 - ps_dict_cross[spec] * bl_hm1 * bl_hm2
                nth_hm2 = ps_dict_auto2[spec] * bl_hm2**2 - ps_dict_cross[spec] * bl_hm1 * bl_hm2
                nth_mean = (nth_hm1 + nth_hm2) / 4
            else:
                nl_hm1 = np.zeros(len(lth))
                nl_hm2 = np.zeros(len(lth))
                nl_mean = np.zeros(len(lth))
            
            nl_sim_mean[spec] += [nth_mean]
                
    for spec in spectra:
        nl_sim_mean[spec] = np.mean(nl_sim_mean[spec], axis=0)

    so_spectra.write_ps(ps_model_dir + "/%s_%s.dat" % (spec_name_noise_mean, name), l, nl_sim_mean, type, spectra=spectra)

