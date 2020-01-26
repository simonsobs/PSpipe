'''
This script compute the null test between TE and ET using ffp10 sims to get an estimate of the errors
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
iStop = 300
freqs = d["freqs"]
lmax = d["lmax"]
lrange = d["lrange"]
bestfit_dir = "best_fits"

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
    
    for fpair in freq_pairs:
            
        f0, f1 = fpair
                
        spec_name = "Planck_%sxPlanck_%s-hm1xhm2" % (f0,f1)
        
        lb, ps_dict = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name, iii), spectra=spectra)

        if f0 != f1:
            spec_name2 = "Planck_%sxPlanck_%s-hm2xhm1" % (f0,f1)
            lb, ps_dict2 = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name2, iii), spectra=spectra)
            ps_dict["TE"] = (ps_dict["TE"] +  ps_dict2["TE"]) / 2
            ps_dict["ET"] = (ps_dict["ET"] +  ps_dict2["ET"]) / 2
        
        sim_null_TE = (ps_dict["TE"] - ps_dict["ET"])


        ps_name = "diff_%sx%s" % (f0, f1)

        lmin_c, lmax_c = lrange[ps_name]

        id=np.where((lb >= lmin_c) & (lb <= lmax_c))

        sim_null_TE = sim_null_TE[id]
        
        vec = np.append(vec, sim_null_TE)
            
        bin_stop += len(sim_null_TE)
        binrange[ps_name] = bin_start, bin_stop
        bin_start = bin_stop

    vec_list += [vec]


mean_vec = np.mean(vec_list, axis=0)
cov = 0
for iii in range(iStart, iStop):
    cov += np.outer(vec_list[iii], vec_list[iii])
cov = cov / (iStop-iStart) - np.outer(mean_vec, mean_vec)



for id_f,fpair in enumerate(freq_pairs):
    
    f0, f1 = fpair
    
    ps_name = "diff_%sx%s" % (f0, f1)
    lmin_c, lmax_c = lrange[ps_name]
        
    lb, ps_dict = so_spectra.read_ps("spectra/spectra_Planck_%sxPlanck_%s-hm1xhm2.dat"%(f0,f1), spectra=spectra)
            
    if f0 != f1:
        lb, ps_dict2 = so_spectra.read_ps("spectra/spectra_Planck_%sxPlanck_%s-hm2xhm1.dat" % (f0,f1), spectra=spectra)
        ps_dict["TE"] = (ps_dict["TE"] +  ps_dict2["TE"]) / 2
        ps_dict["ET"] = (ps_dict["ET"] +  ps_dict2["ET"]) / 2

    data_null_TE = (ps_dict["TE"] - ps_dict["ET"])

    id=np.where((lb >= lmin_c) & (lb <= lmax_c))
        
    lb, data_null_TE = lb[id], data_null_TE[id]

    bin_start, bin_stop = binrange[ps_name]

    std_mean = np.sqrt(cov.diagonal())[bin_start:bin_stop]
        
    fac = lb * (lb + 1) / (2 * np.pi)

    chi2= (np.sum(data_null_TE**2/std_mean**2)/len(lb))
    plt.plot(lb, lb*0)
    plt.errorbar(lb, data_null_TE*lb**2, std_mean*lb**2, fmt = ".", label= "%s, chi2=%.4f" % (ps_name,chi2) )
    plt.legend()
    plt.show()
        

