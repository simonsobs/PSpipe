'''
This script analyze the result of the spectra simulation.
It computes the mean noise power spectrum of the simss.
'''

import numpy as np
import pylab as plt
from pspy import so_dict, so_spectra, pspy_utils
from itertools import combinations_with_replacement as cwr
import sys
import scipy.interpolate
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
    spec_name_noise_mean = "mean_simffp10"

else:
    sim_spectra_dir = "sim_spectra"
    spec_name_noise_mean = "mean_sim"

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


lth = np.arange(2, lmax+2)


for fpair in freq_pairs:
    
    print(fpair)
    
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
                
                lb, nb_hm1 = planck_utils.binning(l, nth_hm1, lmax, size=8)
                lb, nb_hm2 = planck_utils.binning(l, nth_hm2, lmax, size=8)

                nb_mean = (nb_hm1 + nb_hm2) / 4
                nl_interpol_mean = scipy.interpolate.interp1d(lb, nb_mean, fill_value="extrapolate")
                nl_mean = np.array([nl_interpol_mean(i) for i in lth])

            else:
                nl_mean = np.zeros(len(lth))
            
            nl_sim_mean[spec] += [nl_mean]
                
    for spec in spectra:
        nl_sim_mean[spec] = np.mean(nl_sim_mean[spec], axis=0)

    so_spectra.write_ps(ps_model_dir + "/%s_%s_noise.dat" % (spec_name_noise_mean, name), lth, nl_sim_mean, type, spectra=spectra)

