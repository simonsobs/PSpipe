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


# We read each of the covariance matrix element and store then in a dictionnary file

nspec = len(spec_name)
analytic_dict= {}
spectra = ["TT", "TE", "ET", "EE"]
for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        print (name1, name2)
        na, nb = name1.split("x")
        nc, nd = name2.split("x")
        analytic_cov = np.load("%s/analytic_cov_%sx%s_%sx%s.npy" % (cov_dir, na, nb, nc, nd))
        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):
                sub_cov = analytic_cov[s1 * nbins:(s1 + 1) * nbins, s2 * nbins:(s2 + 1) * nbins]
                analytic_dict[sid1, sid2, s1, s2] = sub_cov

# We fill the full covariance matrix with our elements

full_analytic_cov = np.zeros((4 * nspec * nbins, 4 * nspec * nbins))
for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        na, nb = name1.split("x")
        nc, nd = name2.split("x")
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

# for spectra with the same survey and the same array ( sv1 = sv2 and ar1 = ar2) TE=ET
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

np.save("%s/truncated_analytic_cov.npy"%cov_dir, full_analytic_cov)

print ("is matrix positive definite:", data_analysis_utils.is_pos_def(full_analytic_cov))
print ("is matrix symmetric :", data_analysis_utils.is_symmetric(full_analytic_cov))

