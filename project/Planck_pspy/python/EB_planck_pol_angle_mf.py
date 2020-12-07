
import sys
from itertools import combinations_with_replacement as cwr

import numpy as np
from cobaya.run import run
from pspy import pspy_utils, so_dict, so_spectra

import EB_birefringence_tools
import planck_utils


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

binning_file = d["binning_file"]
freqs = d["freqs"]
lmax = d["lmax"]
clfile = d["theoryfile"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
EB_lmin, EB_lmax = d["EB_lmin"], d["EB_lmax"]

spectra_dir = "spectra"
cov_dir = "covariances"
chain_dir = "chains"
pspy_utils.create_directory(chain_dir)




# First create theory array
lth, Clth = pspy_utils.ps_lensed_theory_to_dict(clfile, output_type="Cl", lmax=lmax, start_at_zero=False)
Cb_th = {}
lb, Cb_th["EE"] = planck_utils.binning(lth, Clth["EE"], lmax, binning_file)
lb, Cb_th["BB"]  = planck_utils.binning(lth, Clth["BB"], lmax, binning_file)

id = np.where((lb >= EB_lmin) & (lb <= EB_lmax))
lb, Cb_th["EE"], Cb_th["BB"] = lb[id], Cb_th["EE"][id], Cb_th["BB"][id]
nbins = len(lb)
Cb_th_array = np.zeros((2, nbins))
Cb_th_array[0,:] = Cb_th["EE"]
Cb_th_array[1,:] = Cb_th["BB"]


# Then read the data
cov = np.load("%s/covmat_EB.npy" % cov_dir)
size_cov = cov.shape[0]
cov_EB = cov[np.int(2*size_cov/3):, np.int(2*size_cov/3): ]
inv_cov = np.linalg.inv(cov_EB)

freq_pairs = list(cwr(freqs, 2))
nfreq_pairs= len(freq_pairs)
Cb_data = {}
Cb_data_array = np.zeros((2, nfreq_pairs, nbins))

for id_f, fpair in enumerate(freq_pairs):
    f0, f1 = fpair
    spec_name = "Planck_%sxPlanck_%s-hm1xhm2" % (f0, f1)
    lb, Cb = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name), spectra=spectra)
    if f0 != f1:
        spec_name2 = "Planck_%sxPlanck_%s-hm2xhm1" % (f0, f1)
        lb, Cb2 = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name2), spectra=spectra)
        for spec in ["EE", "EB", "BE", "BB"]:
            Cb[spec] = (Cb[spec] +  Cb2[spec]) / 2
            
    Cb_data["EB", "%sx%s" % (f0, f1)] = (Cb["EB"][id] + Cb["BE"][id]) / 2
    Cb_data_array[0, id_f,  :] = Cb["EE"][id]
    Cb_data_array[1, id_f, :] = Cb["BB"][id]


# Then run the chains
def compute_loglike(alpha100, alpha143, alpha217, beta):
    alpha = {"100": alpha100, "143": alpha143, "217": alpha217}
    vec_res = []
    for id_f, (f0, f1) in enumerate(freq_pairs):
        A = EB_birefringence_tools.get_my_A_vector(alpha[f0], alpha[f1])
        B = EB_birefringence_tools.get_B_vector(alpha[f0], alpha[f1], beta)
        res = (
            Cb_data["EB", "%sx%s" % (f0, f1)]
            - np.dot(A, Cb_data_array[:, id_f, :])
            - np.dot(B, Cb_th_array[:, :])
        )
        vec_res = np.append(vec_res, res)

    chi2 = np.dot(vec_res, np.dot(inv_cov, vec_res))
    return -0.5 * chi2


print("logp(alpha=beta=0) =", compute_loglike(alpha100=0.0, alpha143=0.0, alpha217=0.0, beta=0.0))

info = {
    "likelihood": {"my_like": compute_loglike},
    "params": {
        "alpha100": {"prior": {"min": -5, "max": 5}, "latex": r"\alpha_{100}"},
        "alpha143": {"prior": {"min": -5, "max": 5}, "latex": r"\alpha_{143}"},
        "alpha217": {"prior": {"min": -5, "max": 5}, "latex": r"\alpha_{217}"},
        "beta": {"prior": {"min": -5, "max": 5}, "latex": r"\beta"},
    },
    "sampler": {
        "mcmc": {
            "max_tries": 10 ** 8,
            "Rminus1_stop": 0.001,
            "Rminus1_cl_stop": 0.008,
        }
    },
    "output": "chains/mcmc",
    "force": True,
}

#updated_info, sampler = run(info)
