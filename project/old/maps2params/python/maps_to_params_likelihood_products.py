import numpy as np
import os,sys
from pspy import pspy_utils, so_dict, so_mcm, so_spectra
import maps_to_params_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiments = d["experiments"]
binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]

isims = np.arange(100)

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file,lmax)
n_bins= len(bin_hi)

like_dir = "like_products"
mcm_dir = "mcms"
cov_dir = "covariances"

pspy_utils.create_directory(like_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

g = open("%s/spectra_list.txt" % like_dir, mode="w")

for id_exp1, exp1 in enumerate(experiments):
    freqs1 = d["freqs_%s" % exp1]
    for id_f1, f1 in enumerate(freqs1):
        for id_exp2, exp2 in enumerate(experiments):
            freqs2 = d["freqs_%s" % exp2]
            for id_f2, f2 in enumerate(freqs2):
                
                if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                if  (id_exp1 > id_exp2) : continue
                
                spec_name = "%s_%sx%s_%s" % (exp1, f1, exp2, f2)
                
                for iii in isims:
                    spec_name_cross = "spectra/%s_%s_cross_%05d.dat" % (type, spec_name, iii)
                    l, ps = so_spectra.read_ps(spec_name_cross, spectra=spectra)
                    so_spectra.write_ps("%s/%s_%s_%05d.dat" % (like_dir, type, spec_name, iii),
                                        l,
                                        ps,
                                        type,
                                        spectra=spectra)
            
                prefix = "%s/%s" % (mcm_dir, spec_name)
                
                mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix, spin_pairs=spin_pairs)
                Bbl_TT = Bbl["spin0xspin0"]
                Bbl_TE = Bbl["spin0xspin2"]
                Bbl_EE = Bbl["spin2xspin2"][:Bbl_TE.shape[0],:Bbl_TE.shape[1]]
                
                np.savetxt("%s/Bbl_%s_TT.dat" % (like_dir, spec_name),Bbl_TT)
                np.savetxt("%s/Bbl_%s_TE.dat" % (like_dir, spec_name),Bbl_TE)
                np.savetxt("%s/Bbl_%s_EE.dat" % (like_dir, spec_name),Bbl_EE)
        
            g.write("%s\n" % (spec_name))

g.close()

full_cov = np.load("covariances/full_analytic_cov.npy")
trunc_cov = np.load("covariances/truncated_analytic_cov.npy")

print("full analytic cov is positive definite:", maps_to_params_utils.is_pos_def(full_cov))
print("trunc analytic cov is positive definite:", maps_to_params_utils.is_pos_def(trunc_cov))
print("full analytic cov is symmetric:", maps_to_params_utils.is_symmetric(full_cov))
print("trunc analytic cov is symmetric:", maps_to_params_utils.is_symmetric(trunc_cov))

np.save("%s/full_covariance.npy" % like_dir, full_cov)
np.save("%s/truncated_covariance.npy" % like_dir, trunc_cov)

os.system("cp %s %s/binning.dat" % (binning_file, like_dir))

