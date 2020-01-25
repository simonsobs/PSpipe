'''
This script analyze the result of the spectra simulation
It computes mean spectra and covariance from the sims
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
binning_file = d["binning_file"]
iStart = d["iStart"]
iStop = d["iStop"]
freqs = d["freqs"]
lmax = d["lmax"]
lrange = d["lrange"]
bestfit_dir = "best_fits"
iStop=300

if d["use_ffp10"] == True:
    sim_spectra_dir = "sim_spectra_ffp10"
    mc_dir = "montecarlo_ffp10"
else:
    sim_spectra_dir = "sim_spectra"
    mc_dir = "montecarlo"


pspy_utils.create_directory(mc_dir)

freq_pairs = []
for cross in cwr(freqs, 2):
    freq_pairs += [[cross[0],cross[1]]]


vec_list = []
binrange = {}

for iii in range(iStart,iStop):
    vec = []
    bin_start = 0
    bin_stop = 0
    
    for spec in ["TT", "TE", "EE"]:

        for fpair in freq_pairs:
            
            f0, f1 = fpair
            
            if (spec == "TT") & (f0 == "100") & (f1 == "143"): continue
            if (spec == "TT") & (f0 == "100") & (f1 == "217"): continue

            fname = "%sx%s" % (f0, f1)
        
            spec_name = "Planck_%sxPlanck_%s-hm1xhm2" % (f0,f1)
        
            lb, ps_dict = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name, iii), spectra=spectra)
            
            if spec == "TE":
                ps_dict["TE"] = (ps_dict["TE"] + ps_dict["ET"]) / 2
                if f0 != f1:
                    spec_name2 = "Planck_%sxPlanck_%s-hm2xhm1" % (f0,f1)
                    lb, ps_dict2 = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name2, iii), spectra=spectra)
                    ps_dict2["TE"] = (ps_dict2["TE"] + ps_dict2["ET"]) / 2
        
                    ps_dict["TE"] = (ps_dict["TE"] + ps_dict2["TE"]) / 2


            ps_name = "%s_%sx%s" % (spec, f0, f1)


            lmin_c, lmax_c = lrange[ps_name]

            id=np.where((lb >= lmin_c) & (lb <= lmax_c))

            ps_dict[spec] = ps_dict[spec][id]
        
            vec = np.append(vec, ps_dict[spec])
            
            bin_stop += len(ps_dict[spec])
            binrange[ps_name] = bin_start, bin_stop
            bin_start = bin_stop

    vec_list += [vec]

mean_vec = np.mean(vec_list, axis=0)
cov = 0
for iii in range(iStart, iStop):
    cov += np.outer(vec_list[iii], vec_list[iii])
cov = cov / (iStop-iStart) - np.outer(mean_vec, mean_vec)

np.save("%s/mc_covariance.npy" % (mc_dir), cov)


color_array = ["red", "blue", "orange", "green", "grey", "darkblue"]
for spec in ["TT", "TE", "EE"]:
    for id_f,fpair in enumerate(freq_pairs):
        f0, f1 = fpair
        
        if (spec == "TT") & (f0 == "100") & (f1 == "143"): continue
        if (spec == "TT") & (f0 == "100") & (f1 == "217"): continue
        
        lth, cl_th_and_fg = np.loadtxt("%s/best_fit_%sx%s_%s.dat" % (bestfit_dir, f0, f1, spec), unpack=True)
        lb, cb_th = planck_utils.binning(lth, cl_th_and_fg, lmax, binning_file=binning_file)
        ps_name = "%s_%sx%s" % (spec, f0, f1)
        lmin_c, lmax_c = lrange[ps_name]
        id=np.where((lb >= lmin_c) & (lb <= lmax_c))
        
        lb, cb_th = lb[id], cb_th[id]

        bin_start, bin_stop = binrange[ps_name]
        ps_mean = mean_vec[bin_start:bin_stop]
        std_mean = np.sqrt(cov.diagonal())[bin_start:bin_stop]
        
        fac = lb * (lb + 1) / (2 * np.pi)
        fth = lth * (lth + 1) /  (2 * np.pi)
        
        plt.plot(lb, cb_th/ps_mean)
        
        #plt.errorbar(lb, (cb_th-ps_mean)*fac, std_mean*fac/np.sqrt(iStop-iStart), fmt = ".", label = ps_name)
        #plt.plot(lb, lb * 0)
        #plt.legend()
#plt.show()
         
         #  print (lb.shape, cb_th.shape,ps_mean.shape,std_mean.shape)
         #plt.xlim(0,2200)
         #plt.plot(lth, cl_th_and_fg*fth, color=color_array[id_f], alpha=0.4)
         #plt.plot(lb, cb_th*fac, color=color_array[id_f], alpha=0.4)
         #plt.errorbar(lb, ps_mean*fac, std_mean*fac/np.sqrt(iStop-iStart), fmt = ".",color=color_array[id_f], alpha=0.4)
    plt.show()

