'''
This script analyze the result of the spectra simulation.
'''

import numpy as np
import pylab as plt
from pspy import so_dict, so_spectra, pspy_utils
from itertools import combinations_with_replacement as cwr
import sys
import planck_utils
from scipy.stats.distributions import chi2 as chi2func

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


plot_dir = "plots/sim_analysis/"
spectra_dir = "spectra"

if d["use_ffp10"] == True:
    sim_spectra_dir = "sim_spectra_ffp10"
    mc_dir = "montecarlo_ffp10"
else:
    sim_spectra_dir = "sim_spectra"
    mc_dir = "montecarlo"

pspy_utils.create_directory(mc_dir)
pspy_utils.create_directory(plot_dir)

freq_pairs = []
for cross in cwr(freqs, 2):
    freq_pairs += [[cross[0], cross[1]]]

lmin, lmax = d["EB_lmin"], d["EB_lmax"]

# Create sim cov mat
vec_list_sim = []
for iii in range(iStart, iStop):
    vec_sim = []
    for spec in ["EE", "BB", "EB"]:

        for fpair in freq_pairs:
            f0, f1 = fpair
            spec_name = "Planck_%sxPlanck_%s-hm1xhm2" % (f0,f1)
            lb, ps_dict_sim = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name, iii), spectra=spectra)
            if f0 != f1:
                spec_name2 = "Planck_%sxPlanck_%s-hm2xhm1" % (f0,f1)
                lb, ps_dict_sim2 = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name2, iii), spectra=spectra)
                ps_dict_sim[spec] = (ps_dict_sim[spec] +  ps_dict_sim2[spec]) / 2
                if spec == "EB":
                    ps_dict_sim["BE"] = (ps_dict_sim["BE"] +  ps_dict_sim2["BE"]) / 2
            if spec == "EB":
                ps_dict_sim["EB"] = (ps_dict_sim["EB"] + ps_dict_sim["BE"]) / 2
        
            id = np.where((lb >= lmin) & (lb <= lmax))
            ps_dict_sim[spec] = ps_dict_sim[spec][id]

            vec_sim = np.append(vec_sim, ps_dict_sim[spec])
    vec_list_sim += [vec_sim]

mean_vec_sim = np.mean(vec_list_sim, axis=0)
cov = 0
for iii in range(iStart, iStop):
    cov += np.outer(vec_list_sim[iii], vec_list_sim[iii])
cov = cov / (iStop - iStart) - np.outer(mean_vec_sim, mean_vec_sim)

np.save("%s/mc_covariance_EB.npy" % (mc_dir), cov)

std = np.sqrt(cov.diagonal())

# Data vec

vec_data = []
for spec in ["EE", "BB", "EB"]:
    
    for fpair in freq_pairs:
        f0, f1 = fpair
        spec_name = "Planck_%sxPlanck_%s-hm1xhm2" % (f0,f1)
        lb, ps_dict_data = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name), spectra=spectra)
        if f0 != f1:
            spec_name2 = "Planck_%sxPlanck_%s-hm2xhm1" % (f0,f1)
            lb, ps_dict_data2 = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name2), spectra=spectra)
            ps_dict_data[spec] = (ps_dict_data[spec]+ ps_dict_data2[spec])/2
            if spec == "EB":
                ps_dict_data["BE"] = (ps_dict_data["BE"] +  ps_dict_data2["BE"]) / 2
        if spec == "EB":
            ps_dict_data["EB"] = (ps_dict_data["EB"] + ps_dict_data["BE"]) / 2

        ps_dict_data[spec] = ps_dict_data[spec][id]

        vec_data = np.append(vec_data, ps_dict_data[spec])


lb = lb[id]
n_bins = len(lb)
for nspec, spec in enumerate(["EE", "BB", "EB"]):

    plt.figure(figsize=(20, 15))
    for i, fpair in enumerate(freq_pairs):
        f0, f1 = fpair
        EB = vec_data[6 * nspec * n_bins + i * n_bins: 6 * nspec * n_bins + (i+1) * n_bins]
        std_EB = std[6 * nspec * n_bins + i * n_bins: 6 * nspec * n_bins + (i+1) * n_bins]
        
        plt.subplot(3, 2, i + 1)

        if spec == "EB" :
            chi2 = np.sum(EB**2/std_EB**2)
            PTE = chi2func.sf(chi2, n_bins)
            plt.errorbar(lb, EB, std_EB, fmt=".", label = r"%sx%s, $\chi^{2}$/dof = %0.2f/%d, PTE=%.02f" % (f0, f1, chi2, n_bins, PTE))
        else:
            plt.errorbar(lb, EB, std_EB, fmt=".", label = r"%sx%s" % (f0, f1))

        plt.legend()
    
        np.savetxt("%s/%s_legacy_%sx%s.dat" % (mc_dir, spec, f0, f1), np.transpose(np.array([lb, EB, std_EB])))
    
    plt.savefig("%s.png" % spec, bbox_inches="tight")
    plt.clf()
    plt.close()
