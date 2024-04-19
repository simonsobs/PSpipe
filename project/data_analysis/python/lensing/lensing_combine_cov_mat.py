"""
Combine lensing cov mat generated using amanda codes
Amanda compute the super-sample covariance associated to lensing according to this paper: https://arxiv.org/abs/1401.7992
And the other contribution according to this paper: https://arxiv.org/pdf/1611.01446.pdf, https://arxiv.org/pdf/1205.0474.pdf
Note that the term computing the derivative of lensed wrt unlensed spectra is only relevant for BB
see the discussion leading to equation 14 of : https://arxiv.org/pdf/1205.0474.pdf
the expected folder to look for the lensing cov is "non_gausian_lensing_dir"
"""
import numpy as np
import pylab as plt
from pspy import pspy_utils, so_cov, so_map, so_dict
from pspipe_utils import log, pspipe_list, covariance
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

lmax = d["lmax"]
cov_dir = "covariances"
pspy_utils.create_directory(cov_dir)
non_gausian_lensing_dir = "/global/cfs/cdirs/act/data/tlouis/dr6v4/lensing_non_gaussian/"

binning_file = d["binning_file"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
amanda_spec = ["TT", "TE", "EE", "BB"]

n_spec = len(spectra)

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_hi)

analytic_non_gaussian = np.zeros((n_spec * n_bins, n_spec * n_bins))

for count1, spec1 in enumerate(spectra):
    for count2, spec2 in enumerate(spectra):
        if count1 > count2: continue
        
        if spec1 == "ET": spec1 = "TE"
        if spec2 == "ET": spec2 = "TE"

        if (spec1 in amanda_spec) & (spec2 in amanda_spec):
        
            ssc = np.load(f"{non_gausian_lensing_dir}/test_ssc_unbinned_{spec1.lower()}x{spec2.lower()}.npy")
            deriv_wrt_unlensed = np.load(f"{non_gausian_lensing_dir}/unbinned_{spec1.lower()}x{spec2.lower()}_unlens_derivs.npy")
            deriv_wrt_phi = np.load(f"{non_gausian_lensing_dir}/unbinned_{spec1.lower()}x{spec2.lower()}_phi_derivs.npy")

            if (spec1 == "BB") & (spec2 == "BB"):
                cov = ssc + deriv_wrt_unlensed + deriv_wrt_phi - np.diag(np.diag(deriv_wrt_unlensed))
            else:
                cov = ssc + deriv_wrt_phi
                
            cov_lmax = cov.shape[0] + 2
            
            ll = np.arange(2, cov_lmax)
            
            fac = ll * (ll + 1) / (2 * np.pi)
            
            cov *= np.outer(fac, fac)
            
            analytic_non_gaussian[count1 * n_bins: (count1 + 1) * n_bins, count2 * n_bins: (count2 + 1) * n_bins] = so_cov.bin_mat(cov, binning_file, lmax)

analytic_non_gaussian  = np.triu(analytic_non_gaussian) + np.tril(analytic_non_gaussian.T, -1)

ncovs, na_list, nb_list, nc_list, nd_list = pspipe_list.get_covariances_list(d, delimiter="_")
for i_cov in range(ncovs):
    na, nb, nc, nd = na_list[i_cov], nb_list[i_cov], nc_list[i_cov], nd_list[i_cov]
    np.save(f"{cov_dir}/non_gaussian_lensing_cov_{na}x{nb}_{nc}x{nd}.npy", analytic_non_gaussian)

spec_name_list =  pspipe_list.get_spec_name_list(d, delimiter="_")
x_ar_lensing_cov = covariance.read_cov_block_and_build_full_cov(spec_name_list,
                                                                 cov_dir,
                                                                 "non_gaussian_lensing_cov",
                                                                 spectra_order=spectra,
                                                                 remove_doublon=True,
                                                                 check_pos_def=False)

np.save(f"{cov_dir}/x_ar_non_gaussian_cov_lensing.npy", x_ar_lensing_cov)
