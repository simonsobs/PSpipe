# This script create projection matrix that will be used to combine all arrays, all survey spectra into a set of multifrequency spectra

import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils, so_cov
import data_analysis_utils
from itertools import combinations_with_replacement as cwr
from itertools import product
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
cov_plot_dir = "plots/full_covariance"

surveys = d["surveys"]
lmax = d["lmax"]
binning_file = d["binning_file"]

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

# first let's get a list of all frequencies we plan to study
freq_list = []
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        freq_list += [int(d["nu_eff_%s_%s" % (sv, ar)])]
freq_list = np.sort(list(dict.fromkeys(freq_list)))


# We make a list of all spectra included in the analysis
# We also make one of all spectra with same survey and same array
# This matter for the TE projector since for these spectra TE = ET and
# therefore the ET block have been removed from the cov matrix in order to avoid
# redondancy

spec_name = []
same_spec = []
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                spec_name += ["%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)]
                if (sv1 == sv2) & (ar1 == ar2):
                    same_spec += ["%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)]
                    
n_bins = len(bin_hi)
n_freq= len(freq_list)
n_ps = len(spec_name)
n_ps_same = len(same_spec)


# Now we need to compute the projection matrix
# This bit is extra complicated due to the fact that the TE block part is treated differently
# This happen for two reasons: we don't use any same array, same season ET (since ET = TE)
# and we need to keep T_\nu_1 E_\nu_2 and T_\nu_2 E_\nu_1 separated


# We start with a projector for the TT and EE block

n_cross_freq =  int(n_freq * (n_freq + 1) / 2)
Pmat = np.zeros((n_cross_freq * n_bins, n_ps * n_bins))

cross_freq_list = [ "%sx%s" %(f0,f1) for f0, f1 in cwr(freq_list, 2)]

for c_id, cross_freq in enumerate(cross_freq_list):
    id_start_cf = n_bins * (c_id)
    id_stop_cf = n_bins * (c_id + 1)
    count = 0
    for ps_id, ps in enumerate(spec_name):
        na, nb = ps.split("x")
        nueff_a, nueff_b = d["nu_eff_%s" % na], d["nu_eff_%s" % nb]
        spec_cf_list = ["%sx%s" % (nueff_a, nueff_b), "%sx%s" % (nueff_b, nueff_a)]
        id_start_n = n_bins * (count)
        id_stop_n =  n_bins * (count + 1)
        if cross_freq in spec_cf_list:
            Pmat[id_start_cf:id_stop_cf, id_start_n:id_stop_n] = np.identity(n_bins)
        count += 1

# Now we write a projector for the TE block

n_cross_freq_TE = n_freq ** 2
cross_freq_list = ["%sx%s" % (f0,f1) for f0, f1 in product(freq_list, freq_list)]
print(cross_freq_list)
spec_name_ET = spec_name.copy()
for el in same_spec:
    spec_name_ET.remove(el)
spec_name_TE = np.append(spec_name, spec_name_ET)
n_ps_TE = 2 * n_ps - n_ps_same
Pmat_TE = np.zeros((n_cross_freq_TE * n_bins, n_ps_TE * n_bins))

for c_id, cross_freq in enumerate(cross_freq_list):
    id_start_cf = n_bins * (c_id)
    id_stop_cf = n_bins * (c_id + 1)
    count = 0
    for ps_id, ps in enumerate(spec_name_TE):
        na, nb = ps.split("x")
        nueff_a, nueff_b = d["nu_eff_%s" % na], d["nu_eff_%s" % nb]
        spec_cf_list = ["%sx%s" % (nueff_a, nueff_b)]
        if count >= n_ps:
            # we are in the ET block
            spec_cf_list = ["%sx%s" % (nueff_b, nueff_a)]
            
        id_start_n = n_bins * (count)
        id_stop_n =  n_bins * (count + 1)
        if cross_freq in spec_cf_list:
            Pmat_TE[id_start_cf:id_stop_cf, id_start_n:id_stop_n] = np.identity(n_bins)
            
        count += 1

# We then glue together the TT - TE - EE projectors

shape_x = Pmat.shape[0]
shape_y = Pmat.shape[1]
shape_TE_x = Pmat_TE.shape[0]
shape_TE_y = Pmat_TE.shape[1]

Pmat_tot = np.zeros((2 * shape_x + shape_TE_x, 2 * shape_y + shape_TE_y))
Pmat_tot[:shape_x,:shape_y] = Pmat
Pmat_tot[shape_x: shape_x + shape_TE_x, shape_y: shape_y + shape_TE_y] = Pmat_TE
Pmat_tot[shape_x + shape_TE_x: 2 * shape_x + shape_TE_x, shape_y + shape_TE_y: 2 * shape_y + shape_TE_y] = Pmat

so_cov.plot_cov_matrix(Pmat_tot, file_name="%s/P_mat" % cov_plot_dir)

