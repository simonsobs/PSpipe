import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils, so_spectra, so_cov
from itertools import combinations_with_replacement as cwr
from itertools import product
import data_analysis_utils
import numpy as np
import pylab as plt
import sys, os
from matplotlib.pyplot import cm

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

like_product_dir = "like_product"


surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]

# let's get a list of all frequencies we plan to study
freq_list = []
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        freq_list += [int(d["nu_eff_%s_%s" % (sv, ar)])]
freq_list = np.sort(list(dict.fromkeys(freq_list)))

# Lets read the data vector corresponding to the covariance matrix
 
my_spectra = ["TT", "TE", "EE"]


ell_max_array = [3000, 10000]

Cl_s = {}
sigma_s = {}
compare_with_steve = True
if compare_with_steve:
    l_s, Cl_s["90x90","TT"], sigma_s["90x90","TT"], Cl_s["90x150","TT"], sigma_s["90x150","TT"], Cl_s["150x150","TT"], sigma_s["150x150","TT"] = np.loadtxt("multifreq_spectra/act_dr4.01_multifreq_deep_C_ell_TT.txt", unpack=True)
    l_s, Cl_s["90x90","EE"], sigma_s["90x90","EE"], Cl_s["90x150","EE"], sigma_s["90x150","EE"], Cl_s["150x150","EE"], sigma_s["150x150","EE"] = np.loadtxt("multifreq_spectra/act_dr4.01_multifreq_deep_C_ell_EE.txt", unpack=True)
    l_s, Cl_s["90x90","TE"], sigma_s["90x90","TE"], Cl_s["90x150","TE"], sigma_s["90x150","TE"], Cl_s["150x90","TE"], sigma_s["150x90","TE"],  Cl_s["150x150","TE"], sigma_s["150x150","TE"] = np.loadtxt("multifreq_spectra/act_dr4.01_multifreq_deep_C_ell_TE.txt", unpack=True)
    f_s = l_s * (l_s + 1) / (2*np.pi)


yrange = {}
yrange["TT"] = [30, 5000]
yrange["TE"] = [-150, 150]
yrange["EE"] = [-25, 45]

for ell_max in ell_max_array:
    
    for s1, spec in enumerate(my_spectra):


        plt.figure(figsize=(12, 6))
        if spec == "TE":
            cross_freq_list = ["%sx%s" % (f0,f1) for f0, f1 in product(freq_list, freq_list)]
        else:
            cross_freq_list = ["%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]
        if spec == "TT":
            plt.semilogy()
            
        color=iter(cm.tab10(np.linspace(0,1,10)))

        for cross_freq in cross_freq_list:
            print(cross_freq)
            lb, Db, sigmab = np.loadtxt("%s/spectra_%s_%s.dat" % (like_product_dir, spec, cross_freq), unpack=True)
            id = np.where(lb < ell_max)
            c=next(color)

            plt.errorbar(lb[id], Db[id], sigmab[id], color=c, label = "%s %s" % (spec, cross_freq), fmt=".")

        
        plt.ylim(yrange[spec][0], yrange[spec][1])
        plt.legend()
        plt.savefig("%s/spectra_%s_%d.png" % (like_product_dir, spec, ell_max))
        plt.clf()
        plt.close()


yrange = {}
yrange["TT"] = [30, 5000]
yrange["TE"] = [-150, 150]
yrange["EE"] = [-5, 45]


for ell_max in ell_max_array:

    for s1, spec in enumerate(my_spectra):

        plt.figure(figsize=(12, 6))
        if spec == "TE":
            cross_freq_list = ["%sx%s" % (f0,f1) for f0, f1 in product(freq_list, freq_list)]
        else:
            cross_freq_list = ["%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]
        if spec == "TT":
            plt.semilogy()
            
        color=iter(cm.tab10(np.linspace(0,1,10)))

        for cross_freq in cross_freq_list:
            lb, Db, sigmab = np.loadtxt("%s/spectra_%s_%s.dat" % (like_product_dir, spec, cross_freq), unpack=True)
            id = np.where(lb < ell_max)

            if "220" in cross_freq: continue


            c=next(color)

            if compare_with_steve:
                id_s = np.where(l_s < ell_max)

                plt.errorbar(l_s[id_s], Cl_s[cross_freq, spec][id_s] * f_s[id_s], sigma_s[cross_freq, spec][id_s] * f_s[id_s], color=c, alpha = 0.3, label = "Choi deep %s %s" % (spec, cross_freq))

            plt.errorbar(lb[id], Db[id], sigmab[id], label = "%s %s" % (spec, cross_freq), fmt=".", color=c)

        
        plt.ylim(yrange[spec][0], yrange[spec][1])
        plt.legend()
        plt.savefig("%s/spectra_%s_%d_no220.png" % (like_product_dir, spec, ell_max))
        plt.clf()
        plt.close()



