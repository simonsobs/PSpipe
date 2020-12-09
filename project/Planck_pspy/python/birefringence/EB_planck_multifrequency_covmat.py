"""
This script take the EE-EB-BE-BB covariance elements
and form a multifrequency covariance matrix with form
EE-BB-EB-BE, note that the EB block contain both auto-freq and cross-freq elements
while the BE block only contains cross frequency element (in order to avoid redundancy).
The cov mat is cut at EB_lmin and EB_lmax provided in the global_EB.dict
"""



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
min_planck = d["EB_lmin"]
max_planck = d["EB_lmax"]


bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)

spec_name = []

for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        spec_name += ["%s_%sx%s_%s" % (exp, freq1, exp, freq2)]

# pspy produced EE-EB-BE-BB block covariance matrix, in this step of the code we read them and plug them in
# a multifrequency covariance matrix of size  4 (EE-EB-BE-BB) * nbins * nspec (100x100, 100x143, ..., 353x353)
# we also create a projection matrix that will be used for combining EB and BE in the futur

analytic_dict= {}
spectra = ["EE", "EB", "BE", "BB"]
nspec = len(spec_name)
for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        na, nb = name1.split("x")
        nc, nd = name2.split("x")
        analytic_cov = np.load("%s/analytic_cov_%sx%s_%sx%s_EB.npy" % (cov_dir, na, nb, nc, nd))
        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):
                sub_cov = analytic_cov[s1 * nbins:(s1 + 1) * nbins, s2 * nbins:(s2 + 1) * nbins]
                delta_l = 200
                analytic_dict[sid1, sid2, spec1, spec2] = cut_off_diag_at_delta_l(sub_cov, bin_c, delta_l)


# the order of planck cov mat is EE-BB-EB-BE
spectra = ["EE", "BB", "EB", "BE"]

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
                full_analytic_cov[id_start_1:id_stop_1, id_start_2: id_stop_2] = analytic_dict[sid1, sid2,  spec1, spec2]


# we make the matrix symmetric
transpose = full_analytic_cov.copy().T
transpose[full_analytic_cov != 0] = 0
full_analytic_cov += transpose


# we remove auto frequency BE
block_to_delete = []
for sid, name in enumerate(spec_name):
    na, nb = name.split("x")
    for s, spec in enumerate(spectra):
        _, f0 = na.split("_")
        _, f1 = nb.split("_")
        id_start = sid * nbins + s * nspec * nbins
        id_stop = (sid + 1) * nbins + s * nspec * nbins
        
        if (spec == "BE") & (f0 == f1):
            min_planck = 0
            max_planck = 0
        else:
            min_planck = d["EB_lmin"]
            max_planck = d["EB_lmax"]

        id = np.where(bin_c < min_planck)
        id_start_cut = id_start + len(id[0])
        block_to_delete = np.append(block_to_delete, np.arange(id_start, id_start_cut))

        id = np.where(bin_c > max_planck)
        id_stop_cut = id_stop - (len(id[0]))
        block_to_delete = np.append(block_to_delete, np.arange(id_stop_cut,id_stop))
        nbins_rm =  (id_stop - id_stop_cut) + (id_start_cut - id_start)
        print("for %s %sx%s remove %d bin of %d" % (spec, f0, f1, nbins_rm, nbins))
        
block_to_delete = block_to_delete.astype(int)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=1)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=0)

print ("is matrix positive definite:", planck_utils.is_pos_def(full_analytic_cov))
print ("is matrix symmetric :", planck_utils.is_symmetric(full_analytic_cov))

#inv_full_analytic_cov = np.linalg.inv(full_analytic_cov)
np.save("%s/covmat_EB.npy" % cov_dir, full_analytic_cov)

