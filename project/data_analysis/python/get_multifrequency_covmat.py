# This script use the covariance matrix elements to form a multifrequency covariance matrix
# with block TT - TE - ET - EE
# Note that for the ET block, we do not include any same array, same survey spectra, since for
# these guys TE = ET

import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils
import data_analysis_utils
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
mc_dir = "montecarlo"
cov_plot_dir = "plots/full_covariance"

pspy_utils.create_directory(cov_plot_dir)

surveys = d["surveys"]
lmax = d["lmax"]
binning_file = d["binning_file"]
multistep_path = d["multistep_path"]

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)

# We make a list of all spectra included in the analysis

spec_name = []
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                spec_name += ["%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)]


# We read each of the covariance matrix element associated to each spectra pair
# and store then in a dictionnary file

nspec = len(spec_name)
analytic_dict= {}
spectra = ["TT", "TE", "ET", "EE"]
for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        print (name1, name2)
        analytic_cov = np.load("%s/analytic_cov_%s_%s.npy" % (cov_dir, name1, name2))
        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):
                sub_cov = analytic_cov[s1 * nbins:(s1 + 1) * nbins, s2 * nbins:(s2 + 1) * nbins]
                analytic_dict[sid1, sid2, s1, s2] = sub_cov

# We fill the full covariance matrix with our elements
# The cov mat format is [block TT, block TE, block ET, block EE]
# the block contain all cross array and season spectra

full_analytic_cov = np.zeros((4 * nspec * nbins, 4 * nspec * nbins))
for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):
                id_start_1 = sid1 * nbins + s1 * nspec * nbins
                id_stop_1 = (sid1 + 1) * nbins + s1 * nspec * nbins
                id_start_2 = sid2 * nbins + s2 * nspec * nbins
                id_stop_2 = (sid2 + 1) * nbins + s2 * nspec * nbins
                full_analytic_cov[id_start_1:id_stop_1, id_start_2: id_stop_2] = analytic_dict[sid1, sid2, s1, s2]

# We make the matrix symmetric and save it

transpose = full_analytic_cov.copy().T
transpose[full_analytic_cov != 0] = 0
full_analytic_cov += transpose
np.save("%s/full_analytic_cov.npy"%cov_dir, full_analytic_cov)


# for spectra with the same survey and the same array (sv1 = sv2 and ar1 = ar2) TE = ET
# therefore we remove the ET block in order for the covariance not to be redondant

block_to_delete = []
for sid, name in enumerate(spec_name):
    na, nb = name.split("x")
    for s, spec in enumerate(spectra):
        id_start = sid * nbins + s * nspec * nbins
        id_stop = (sid + 1) * nbins + s * nspec * nbins
        if (na == nb) & (spec == "ET"):
            block_to_delete = np.append(block_to_delete, np.arange(id_start, id_stop))

block_to_delete = block_to_delete.astype(int)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=1)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=0)

np.save("%s/truncated_analytic_cov.npy" % cov_dir, full_analytic_cov)

print ("is matrix positive definite:", data_analysis_utils.is_pos_def(full_analytic_cov))
print ("is matrix symmetric :", data_analysis_utils.is_symmetric(full_analytic_cov))


# This part compare the analytic covariance with the montecarlo covariance
# In particular it produce plot of all diagonals of the matrix with MC vs analytics
# We use our usual javascript visualisation tools

full_mc_cov = np.load("%s/cov_restricted_all_cross.npy"%mc_dir)


os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, cov_plot_dir))
file = "%s/covariance.html" % (cov_plot_dir)
g = open(file, mode="w")
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> covariance </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
g.write('<style> \n')
g.write('body { text-align: center; } \n')
g.write('img { width: 100%; max-width: 1200px; } \n')
g.write('</style> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub>\n')

size=int(full_analytic_cov.shape[0]/nbins)
count=0
for ispec in range(-size+1, size):
    
    rows, cols = np.indices(full_mc_cov.shape)
    row_vals = np.diag(rows, k=ispec*nbins)
    col_vals = np.diag(cols, k=ispec*nbins)
    mat = np.ones(full_mc_cov.shape)
    mat[row_vals, col_vals] = 0
    
    str = "cov_diagonal_%03d.png" % (count)

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.plot(np.log(np.abs(full_analytic_cov.diagonal(ispec*nbins))))
    plt.plot(np.log(np.abs(full_mc_cov.diagonal(ispec*nbins))), '.')
    plt.legend()
    plt.subplot(1,2,2)
    plt.imshow(np.log(np.abs(full_analytic_cov*mat)))
    plt.savefig("%s/%s"%(cov_plot_dir,str))
    plt.clf()
    plt.close()
    
    g.write('<div class=sub>\n')
    g.write('<img src="'+str+'"  /> \n')
    g.write('</div>\n')
    
    count+=1

g.write('</body> \n')
g.write('</html> \n')
g.close()


