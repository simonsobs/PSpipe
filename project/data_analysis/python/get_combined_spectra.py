import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils, so_spectra, so_cov
from itertools import combinations_with_replacement as cwr
from itertools import product
import data_analysis_utils
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
spec_dir = "spectra"
like_product_dir = "like_product"

pspy_utils.create_directory(like_product_dir)

surveys = d["surveys"]
type = d["type"]
lmax = d["lmax"]
binning_file = d["binning_file"]

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

# Lets read the data vector corresponding to the covariance matrix
 
data_vec = []
print("read data vec") 
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
                    
                    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
                    lb, Db = so_spectra.read_ps(spec_dir + "/%s_%s_cross.dat" % (type, spec_name), spectra=spectra)
                    # remove same array, same season ET
                    if (spec == "ET") & (ar1 == ar2) & (sv1 == sv2): continue
                    data_vec = np.append(data_vec, Db[spec])

# Lets combine the data (following the doc)

print("invert cov mat")
inv_cov_mat = np.linalg.inv(cov_mat)

proj_cov_mat = np.linalg.inv(np.dot(np.dot(P_mat, inv_cov_mat), P_mat.T))
proj_data_vec = np.dot(proj_cov_mat, np.dot(P_mat, np.dot(inv_cov_mat, data_vec)))

print ("is matrix positive definite:", data_analysis_utils.is_pos_def(proj_cov_mat))
print ("is matrix symmetric :", data_analysis_utils.is_symmetric(proj_cov_mat))
#so_cov.plot_cov_matrix(np.log(proj_cov_mat))

np.save("%s/combined_analytic_cov.npy" % like_product_dir, proj_cov_mat)
np.savetxt("%s/data_vec.dat" % like_product_dir, proj_data_vec)


my_spectra = ["TT", "TE", "EE"]
yrange = {}
yrange["TT"] = [30, 5000]
yrange["TE"] = [-150, 150]
yrange["EE"] = [-25, 50]

count = 0
for s1, spec in enumerate(my_spectra):
    
    plt.figure(figsize=(12, 6))
    if spec == "TE":
        cross_freq_list = ["%sx%s" % (f0,f1) for f0, f1 in product(freq_list, freq_list)]
    else:
        cross_freq_list = ["%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]
    if spec == "TT":
        plt.semilogy()
    for cross_freq in cross_freq_list:
        
        Db = proj_data_vec[count * n_bins: (count + 1) * n_bins]
        sigmab = np.sqrt(proj_cov_mat.diagonal()[count * n_bins: (count + 1) * n_bins])

        np.savetxt("%s/spectra_%s_%s.dat" % (like_product_dir, spec, cross_freq), np.transpose([lb, Db, sigmab]))
        
        plt.errorbar(lb, Db, sigmab, label = "%s %s" % (spec, cross_freq), fmt=".")

        count += 1
        
    plt.ylim(yrange[spec][0], yrange[spec][1])
    plt.legend()
    plt.savefig("%s/spectra_%s.png" % (like_product_dir, spec))
    plt.clf()
    plt.close()

count = 0
for s1, spec in enumerate(my_spectra):

    plt.figure(figsize=(12, 6))
    if spec == "TE":
        cross_freq_list = ["%sx%s" % (f0,f1) for f0, f1 in product(freq_list, freq_list)]
    else:
        cross_freq_list = ["%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]
    if spec == "TT":
        plt.semilogy()
    for cross_freq in cross_freq_list:
        

        Db = proj_data_vec[count * n_bins: (count + 1) * n_bins]
        sigmab = np.sqrt(proj_cov_mat.diagonal()[count * n_bins: (count + 1) * n_bins])
        
        count += 1
        if "220" in cross_freq: continue

        np.savetxt("%s/spectra_%s_%s.dat" % (like_product_dir, spec, cross_freq), np.transpose([lb, Db, sigmab]))
        
        plt.errorbar(lb, Db, sigmab, label = "%s %s" % (spec, cross_freq), fmt=".")

        
    plt.ylim(yrange[spec][0], yrange[spec][1])
    plt.legend()
    plt.savefig("%s/spectra_%s_no220.png" % (like_product_dir, spec))
    plt.clf()
    plt.close()



