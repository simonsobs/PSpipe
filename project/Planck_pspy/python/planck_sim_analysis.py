'''
This script analyze the result of the spectra simulation.
It compares the dispersion of monte-carlo simulation with planck analytical covariance matrix
'''
import matplotlib
matplotlib.use("Agg")
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


plot_dir = "plots/sim_analysis/"
cov_dir = "covariances"

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

vec_list = []
bin_range = {}

for iii in range(iStart, iStop):
    vec = []
    bin_start = 0
    bin_stop = 0
    
    for spec in ["TT", "EE", "TE"]:

        for fpair in freq_pairs:
            
            f0, f1 = fpair
            
            if (spec == "TT") & (f0 == "100") & (f1 == "143"): continue
            if (spec == "TT") & (f0 == "100") & (f1 == "217"): continue
        
            spec_name = "Planck_%sxPlanck_%s-hm1xhm2" % (f0,f1)
            
            lb, ps_dict = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name, iii), spectra=spectra)
            
            if f0 != f1:
                spec_name2 = "Planck_%sxPlanck_%s-hm2xhm1" % (f0,f1)
                lb, ps_dict2 = so_spectra.read_ps("%s/sim_spectra_%s_%04d.dat" % (sim_spectra_dir, spec_name2, iii), spectra=spectra)
                ps_dict[spec] = (ps_dict[spec] +  ps_dict2[spec]) / 2
                if spec == "TE":
                    ps_dict["ET"] = (ps_dict["ET"] +  ps_dict2["ET"]) / 2

            if spec == "TE":
                ps_dict["TE"] = (ps_dict["TE"] + ps_dict["ET"]) / 2
        
            ps_name = "%s_%sx%s" % (spec, f0, f1)

            lmin_c, lmax_c = lrange[ps_name]

            id=np.where((lb >= lmin_c) & (lb <= lmax_c))

            ps_dict[spec] = ps_dict[spec][id]
        
            vec = np.append(vec, ps_dict[spec])
            
            bin_stop += len(ps_dict[spec])
            bin_range[ps_name] = bin_start, bin_stop
            bin_start = bin_stop

    vec_list += [vec]

mean_vec = np.mean(vec_list, axis=0)

cov = 0
for iii in range(iStart, iStop):
    cov += np.outer(vec_list[iii], vec_list[iii])
cov = cov / (iStop - iStart) - np.outer(mean_vec, mean_vec)

np.save("%s/mc_covariance.npy" % (mc_dir), cov)

inv_cov_planck = np.load("data/planck_data/covmat.npy")
cov_planck = np.linalg.inv(inv_cov_planck)
variance_planck = cov_planck.diagonal()

inv_cov_analytic = np.load("%s/inv_covmat.npy" % cov_dir)
cov_analytic = np.linalg.inv(inv_cov_analytic)
variance_analytic = cov_analytic.diagonal()



color_array = ["red", "blue", "orange", "green", "grey", "darkblue"]

id_low_planck = 0
id_high_planck = 0

for spec in ["TT", "EE", "TE"]:
    for id_f, fpair in enumerate(freq_pairs):
        
        f0, f1 = fpair
        
        if (spec == "TT") & (f0 == "100") & (f1 == "143"): continue
        if (spec == "TT") & (f0 == "100") & (f1 == "217"): continue
        
        l_planck, _, _ = np.loadtxt("data/planck_data/spectrum_%s_%sx%s.dat"%(spec, f0, f1), unpack=True)
        id_high_planck += len(l_planck)
        std_planck = np.sqrt(variance_planck)[id_low_planck:id_high_planck]
        std_analytic = np.sqrt(variance_analytic)[id_low_planck:id_high_planck]

        lth, cl_th_and_fg = np.loadtxt("%s/best_fit_%sx%s_%s.dat" % (bestfit_dir, f0, f1, spec), unpack=True)
        lb, cb = planck_utils.binning(lth, cl_th_and_fg, lmax, binning_file=binning_file)
        ps_name = "%s_%sx%s" % (spec, f0, f1)
        lmin_c, lmax_c = lrange[ps_name]
        id=np.where((lb >= lmin_c) & (lb <= lmax_c))
        lb, cb = lb[id], cb[id]
        fac = lb * (lb + 1) / (2 * np.pi)

        bin_start, bin_stop = bin_range[ps_name]
        std_sim = np.sqrt(cov.diagonal())[bin_start:bin_stop]
        mean_sim = mean_vec[bin_start:bin_stop]
        
        plt.figure(figsize=(12, 8))
        plt.errorbar(lb, cb * fac, color="grey", label = "input theory")
        plt.errorbar(lb, mean_sim * fac, std_sim * fac, color="red", fmt = "." , label = "mean %s %s GHzx %s GHz"%(spec, f0, f1), alpha=0.4)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.ylabel(r"$D_\ell$", fontsize=20)
        plt.legend()
        plt.savefig("%s/mean_spectra_%s_%sx%s.png" % (plot_dir, spec, f0, f1))
        plt.clf()
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.semilogy()
        plt.errorbar(l_planck, std_planck, color="grey", label = "std planck")
        plt.errorbar(l_planck, std_analytic, color="blue", label = "std std_analytic")
        plt.errorbar(lb, std_sim, color="red", fmt = ".", label = "std sim %s %s GHzx %s GHz"%(spec, f0, f1), alpha=0.4)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.ylabel(r"$\sigma_\ell$", fontsize=20)
        plt.legend()
        plt.savefig("%s/std_%s_%sx%s.png" % (plot_dir, spec, f0, f1))
        plt.clf()
        plt.close()

        id_low_planck = id_high_planck
    
