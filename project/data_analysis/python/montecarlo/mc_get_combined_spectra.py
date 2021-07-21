#PRELIM

import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils, so_spectra, so_cov, so_mcm
from itertools import combinations_with_replacement as cwr
from itertools import product
import data_analysis_utils
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
spec_dir = "sim_spectra"
like_product_dir = "sim_like_product"
mcm_dir = "mcms"
bestfit_dir = "best_fits"


pspy_utils.create_directory(like_product_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]
iStart = d["iStart"]
iStop = d["iStop"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]


bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_hi)

# we will need the covariance matrix and the projection matrix
cov_mat = np.load("%s/truncated_analytic_cov.npy" % cov_dir)
P_mat = np.load("%s/projection_matrix.npy" % cov_dir)

# let's get a list of all frequencies we plan to study
freq_list = []
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        freq_list += [int(d["nu_eff_%s_%s" % (sv, ar)])]
freq_list = np.sort(list(dict.fromkeys(freq_list)))


# Lets combine the data (following the doc)

print("invert cov mat")
inv_cov_mat = np.linalg.inv(cov_mat)

proj_cov_mat = np.linalg.inv(np.dot(np.dot(P_mat, inv_cov_mat), P_mat.T))
print ("is matrix positive definite:", data_analysis_utils.is_pos_def(proj_cov_mat))
print ("is matrix symmetric :", data_analysis_utils.is_symmetric(proj_cov_mat))
so_cov.plot_cov_matrix(np.log(proj_cov_mat), file_name = "%s/combined_analytic_cov" % like_product_dir)
np.save("%s/combined_analytic_cov.npy" % like_product_dir, proj_cov_mat)



all_vec = []
for iii in range(iStart, iStop):
    data_vec = []
    my_spectra = ["TT", "TE", "ET", "EE"]
    for spec in my_spectra:
        for id_sv1, sv1 in enumerate(surveys):
            arrays_1 = d["arrays_%s" % sv1]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = d["arrays_%s" % sv2]
                    for id_ar2, ar2 in enumerate(arrays_2):
                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        if  (id_sv1 > id_sv2) : continue
                
                        spec_name = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                        
                        lb, Db = so_spectra.read_ps(spec_dir + "/%s_%s_cross_%05d.dat" % (type, spec_name, iii), spectra=spectra)
                        # remove same array, same season ET
                        if (spec == "ET") & (ar1 == ar2) & (sv1 == sv2): continue
                        data_vec = np.append(data_vec, Db[spec])

    proj_data_vec = np.dot(proj_cov_mat, np.dot(P_mat, np.dot(inv_cov_mat, data_vec)))
    np.savetxt("%s/data_vec_%05d.dat" % (like_product_dir, iii), proj_data_vec)
    all_vec += [ proj_data_vec]



proj_data_vec_mean = np.mean(all_vec, axis=0)

mc_proj_cov_mat = 0
for iii in range(iStart, iStop):
    mc_proj_cov_mat += np.outer(all_vec[iii], all_vec[iii])

mc_proj_cov_mat = mc_proj_cov_mat / (iStop-iStart) - np.outer(proj_data_vec_mean, proj_data_vec_mean)
so_cov.plot_cov_matrix(np.log(mc_proj_cov_mat), file_name = "%s/combined_mc_cov" % like_product_dir)
np.save("%s/combined_mc_cov.npy" % like_product_dir, proj_cov_mat)

diag_cov = proj_cov_mat.diagonal()
mc_diag_cov = mc_proj_cov_mat.diagonal()


plt.figure(figsize=(12, 6))
plt.semilogy()
plt.plot(np.sqrt(diag_cov))
plt.plot(np.sqrt(mc_diag_cov),".")
plt.legend()
plt.savefig("%s/combined_diagonal.png" % (like_product_dir))
plt.clf()
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(np.sqrt(diag_cov) * 0 + 1)
plt.plot(np.sqrt(mc_diag_cov)/np.sqrt(diag_cov))
plt.savefig("%s/combined_diagonal_ratio_mc_over_analytic.png" % (like_product_dir))
plt.clf()
plt.close()



# Now compare combined spectra with theory
from matplotlib.pyplot import cm

clfile = "%s/lcdm.dat" % bestfit_dir
lth, Dlth = pspy_utils.ps_lensed_theory_to_dict(clfile, output_type=type, lmax=lmax, start_at_zero=False)

my_spectra = ["TT", "TE", "EE"]
count = 0
for s1, spec in enumerate(my_spectra):
    plt.figure(figsize=(12, 6))

    if spec == "TE":
        cross_freq_list = ["%sx%s" % (f0,f1) for f0, f1 in product(freq_list, freq_list)]
    else:
        cross_freq_list = ["%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]

    color=iter(cm.tab20(np.linspace(0, 1, 10)))

    for cross_freq in cross_freq_list:
        ps = Dlth[spec].copy()

        if spec == "TT":
            plt.semilogy()
            f0, f1 = cross_freq.split("x")
            _, flth = np.loadtxt("%s/fg_%sx%s_TT.dat" %(bestfit_dir, f0, f1), unpack=True)
            ps += flth[:lmax]

        
        Db = proj_data_vec_mean[count * n_bins: (count + 1) * n_bins]
        sigmab = np.sqrt(proj_cov_mat.diagonal()[count * n_bins: (count + 1) * n_bins])

        c=next(color)

        plt.errorbar(lb, Db, sigmab, label = "%s %s" % (spec, cross_freq), fmt=".", color=c)
        plt.errorbar(lth, ps, label = "theory %s %s" % (spec, cross_freq), color=c)

        count += 1
    
    plt.legend()
    plt.savefig("%s/mc_spectra_%s.png" % (like_product_dir, spec))
    plt.clf()
    plt.close()
