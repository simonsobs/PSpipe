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

# Lets combine the cross array spectra in cross freq spectra (following the doc)
# we will need the covariance matrix and the projection matrix

cov_mat = np.load("%s/truncated_analytic_cov.npy" % cov_dir)
P_mat = np.load("%s/projection_matrix_x_ar_to_x_freq.npy" % cov_dir)


proj_cov_mat, proj_data_vec = pspy_utils.maximum_likelihood_combination(cov_mat,
                                                                        P_mat,
                                                                        data_vec,
                                                                        test_matrix=True)

np.savetxt("%s/data_vec.dat" % like_product_dir, proj_data_vec)
np.save("%s/combined_analytic_cov.npy" % like_product_dir, proj_cov_mat)

# this is a test: force the covariance matrix to be symmetric
proj_cov_mat_forcesim = np.tril(proj_cov_mat) + np.triu(proj_cov_mat.T, 1)


np.save("%s/combined_analytic_cov_forcesim.npy" % like_product_dir, proj_cov_mat_forcesim)
print(np.max(proj_cov_mat-proj_cov_mat_forcesim))

my_spectra = ["TT", "TE", "EE"]

count = 0
for s1, spec in enumerate(my_spectra):

    if spec == "TE":
        cross_freq_list = ["%sx%s" % (f0,f1) for f0, f1 in product(freq_list, freq_list)]
    else:
        cross_freq_list = ["%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]
            
    for cross_freq in cross_freq_list:
    
        Db = proj_data_vec[count * n_bins: (count + 1) * n_bins]
        sigmab = np.sqrt(proj_cov_mat.diagonal()[count * n_bins: (count + 1) * n_bins])

        np.savetxt("%s/spectra_%s_%s.dat" % (like_product_dir, spec, cross_freq), np.transpose([lb, Db, sigmab]))
        count += 1



# Now let's combine all cross freq TE and EE into a single final TE and EE power spectrum
# for this we need the final projection matrix

P_mat_final = np.load("%s/projection_matrix_x_freq_to_final.npy" % cov_dir)
final_cov_mat, final_data_vec = pspy_utils.maximum_likelihood_combination(proj_cov_mat,
                                                                          P_mat_final,
                                                                          proj_data_vec,
                                                                          test_matrix=True)


np.savetxt("%s/data_vec_final.dat" % like_product_dir, final_data_vec)
np.save("%s/final_analytic_cov.npy" % like_product_dir, final_cov_mat)

count = 0
for s1, spec in enumerate(my_spectra):
    if spec == "TT":
        cross_freq_list = ["%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]
    else:
        cross_freq_list = ["all"]

    for cross_freq in cross_freq_list:
    
        Db = final_data_vec[count * n_bins: (count + 1) * n_bins]
        sigmab = np.sqrt(final_cov_mat.diagonal()[count * n_bins: (count + 1) * n_bins])

        np.savetxt("%s/final_spectra_%s_%s.dat" % (like_product_dir, spec, cross_freq), np.transpose([lb, Db, sigmab]))
        count += 1

