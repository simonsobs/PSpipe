from pspy import so_dict, pspy_utils
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiments = d["experiments"]
lmax = d["lmax"]
binning_file = d["binning_file"]

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nb = len(bin_hi)

spec_name = []

for id_exp1, exp1 in enumerate(experiments):
    freqs1 = d["freqs_%s" % exp1]
    for id_f1, f1 in enumerate(freqs1):
        for id_exp2, exp2 in enumerate(experiments):
            freqs2 = d["freqs_%s" % exp2]
            for id_f2, f2 in enumerate(freqs2):
                if  (id_exp1 == id_exp2) & (id_f1 >id_f2) : continue
                if  (id_exp1 > id_exp2) : continue
                spec_name += ["%s_%sx%s_%s" % (exp1, f1, exp2, f2)]

analytic_dict= {}
spectra = ["TT", "TE", "ET", "EE"]

nspec = len(spec_name)

for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1 > sid2: continue
        print (spec1,spec2)
        na, nb = spec1.split("x")
        nc, nd = spec2.split("x")

        analytic_cov = np.load("%s/analytic_cov_%sx%s_%sx%s.npy" % (cov_dir, na, nb, nc, nd))
        
        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):
                
                sub_cov = analytic_cov[s1 * nb:(s1 + 1) * nb, s2 * nb:(s2 + 1) * nb]
                analytic_dict[sid1, sid2, s1, s2] = sub_cov
                analytic_dict[sid2, sid1, s1, s2] = sub_cov.T


full_cov = np.zeros((4 * nspec * nb, 4 * nspec * nb))

for s1, spec1 in enumerate(spectra):
    for sid1, name1 in enumerate(spec_name):
        id_start_1 = sid1 * nb + s1 * nspec * nb
        id_stop_1 = (sid1 + 1) * nb + s1 * nspec * nb
        for s2, spec2 in enumerate(spectra):
            for sid2, name2 in enumerate(spec_name):
                id_start_2 = sid2 * nb + s2 * nspec * nb
                id_stop_2 = (sid2 + 1) * nb + s2 * nspec * nb
                print (name1, spec1, name2, spec2, id_start_1, id_stop_1, id_start_2, id_stop_2)


