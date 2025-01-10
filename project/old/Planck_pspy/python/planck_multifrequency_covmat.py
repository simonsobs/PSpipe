#import matplotlib
#matplotlib.use("Agg")
from pspy import so_dict, pspy_utils, so_cov
import planck_utils
import numpy as np
import pylab as plt
import sys, os

def cut_off_diag_at_delta_l(mat, lb, delta_l):
    cut_mat = np.zeros(mat.shape)
    nbins = mat.shape[0]
    for i in range(nbins):
        for j in range(nbins):
            if np.abs(lb[i]-lb[j]) < delta_l:
                cut_mat[i,j] = mat[i,j]
    return cut_mat

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
mc_dir = "montecarlo"
exp = "Planck"
lmax = d["lmax"]
binning_file = d["binning_file"]
freqs = d["freqs"]

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)

spec_name = []

for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        spec_name += ["%s_%sx%s_%s" % (exp, freq1, exp, freq2)]

# pspy produced TT-TE-ET-EE block covariance matrix, in this step of the code we read them and plug them in
# a multifrequency covariance matrix of size  4 (TT-TE-ET-EE) * nbins * nspec (100x100, 100x143, ..., 217x217)
# we also create a projection matrix that will be used for combining TE and ET in the futur

analytic_dict= {}
spectra = ["TT", "TE", "ET", "EE"]
nspec = len(spec_name)
for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        na, nb = name1.split("x")
        nc, nd = name2.split("x")
        analytic_cov = np.load("%s/analytic_cov_%sx%s_%sx%s.npy" % (cov_dir, na, nb, nc, nd))
        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):
                sub_cov = analytic_cov[s1 * nbins:(s1 + 1) * nbins, s2 * nbins:(s2 + 1) * nbins]
                delta_l = 200
                analytic_dict[sid1, sid2, spec1, spec2] = cut_off_diag_at_delta_l(sub_cov, bin_c, delta_l)


# the order of planck cov mat is TT-EE-TE
spectra = ["TT", "EE", "TE", "ET"]

full_analytic_cov = np.zeros((4 * nspec * nbins, 4 * nspec * nbins))
Pmat = np.zeros((4 * nspec * nbins, 4 * nspec * nbins))

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
                full_analytic_cov[id_start_1:id_stop_1, id_start_2: id_stop_2] = analytic_dict[sid1, sid2,  spec1, spec2]

                if name1 == name2:
                    _, f0 = na.split("_")
                    _, f1 = nb.split("_")

                    if spec1 == spec2:
                        Pmat[id_start_1:id_stop_1, id_start_2: id_stop_2] = np.eye(nbins)
                    if f0 != f1:
                        if spec1 == "ET" or spec1 == "TE":
                            if spec2 == "ET" or spec2 == "TE":
                                Pmat[id_start_1:id_stop_1, id_start_2: id_stop_2] = np.eye(nbins)/2

# we make the matrix symmetric
transpose = full_analytic_cov.copy().T
transpose[full_analytic_cov != 0] = 0
full_analytic_cov += transpose


# we remove auto frequency ET
block_to_delete = []
for sid, name in enumerate(spec_name):
    na, nb = name.split("x")
    for s, spec in enumerate(spectra):
        _, f0 = na.split("_")
        _, f1 = nb.split("_")
        id_start = sid * nbins + s * nspec * nbins
        id_stop = (sid + 1) * nbins + s * nspec * nbins
        if (spec == "ET") & (f0 == f1):
            block_to_delete = np.append(block_to_delete, np.arange(id_start, id_stop))


block_to_delete = block_to_delete.astype(int)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=1)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=0)
Pmat = np.delete(Pmat, block_to_delete, axis=1)
Pmat = np.delete(Pmat, block_to_delete, axis=0)
Pmat = Pmat[0: Pmat.shape[0], 0: 3 * nspec * nbins]

full_analytic_cov = (np.dot(np.dot(Pmat.T, full_analytic_cov), Pmat))


spectra = ["TT", "EE", "TE"]

block_to_delete = []

for sid, name in enumerate(spec_name):
    na, nb = name.split("x")
    for s, spec in enumerate(spectra):
        
        _, f0 = na.split("_")
        _, f1 = nb.split("_")
        
        print (f0, f1, spec)
        id_start = sid * nbins + s * nspec * nbins
        id_stop = (sid + 1) * nbins + s * nspec * nbins
        
        # Remove spectra and bin following planck legacy
        if (spec == "TT") & (f0 == "100") & (f1 == "143"):
            block_to_delete = np.append(block_to_delete, np.arange(id_start, id_stop))
        elif (spec == "TT") & (f0 == "100") & (f1 == "217"):
            block_to_delete = np.append(block_to_delete, np.arange(id_start, id_stop))

        else:
            l_planck, _, _ = np.loadtxt("data/planck_data/spectrum_%s_%sx%s.dat"%(spec, f0, f1), unpack=True)
            min_planck = np.min(l_planck)
            max_planck = np.max(l_planck)
            
            id = np.where(bin_c < min_planck)
            id_start_cut = id_start + len(id[0])
            block_to_delete = np.append(block_to_delete, np.arange(id_start, id_start_cut))
            
            id = np.where(bin_c > max_planck)
            id_stop_cut = id_stop - (len(id[0]))
            block_to_delete = np.append(block_to_delete, np.arange(id_stop_cut,id_stop))

block_to_delete = block_to_delete.astype(int)


full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=1)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=0)

print ("is matrix positive definite:", planck_utils.is_pos_def(full_analytic_cov))
print ("is matrix symmetric :", planck_utils.is_symmetric(full_analytic_cov))

inv_full_analytic_cov = np.linalg.inv(full_analytic_cov)
np.save("%s/inv_covmat.npy"%cov_dir, inv_full_analytic_cov)

