import numpy as np
import scipy.stats as stats

ell = np.arange(4001)

bin_edges = np.arange(0, 5000, 50)

lmax = bin_edges[1:].copy()
lmin = bin_edges[:-1].copy()
lmin[0]= 2
lmin[1:] += 1

lmean = (lmin + lmax)/2

np.savetxt("binning_spt.dat", np.transpose([lmin, lmax, lmean]))

