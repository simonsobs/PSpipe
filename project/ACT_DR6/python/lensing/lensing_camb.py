"""
Precompute the lensed and unlensed Cls and store them in a covariance form that will allow map simulation
Previously this was done in the simulation script to avoid having lot of script, but there is an incompatibility
between CAMB openmp and pspy openmp at nersc
"""
import numpy as np
import pylab as plt
from pspy import pspy_utils, so_dict, so_spectra
from pspipe_utils import log
import sys


def get_unlensed_ps_mat(l, ps_unlensed):
    assert(l[0] == 0), "you want ps_unlensed to start at 0"
    ps_mat = np.zeros((4, 4, len(l)))
    for c1, f1 in enumerate("PTEB"):
        for c2, f2 in enumerate("PTEB"):
            ps_mat[c1, c2, :] = ps_unlensed[f1+f2]
    return ps_mat
    
def get_ps_mat(l, ps_lensed):
    assert(l[0] == 0), "you want ps_lensed to start at 0 "
    ps_mat = np.zeros((3, 3, len(l)))
    for c1, f1 in enumerate("TEB"):
        for c2, f2 in enumerate("TEB"):
            ps_mat[c1, c2, :] = ps_lensed[f1+f2]

    return ps_mat
    
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

lmax_sim = d["lmax"]
cosmo_params = d["cosmo_params"]
accuracy_pars = d["accuracy_params"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lensing_dir = "lensing"
pspy_utils.create_directory(lensing_dir)

l, ps_lensed = pspy_utils.ps_from_params(cosmo_params, "Cl", lmax_sim, start_at_zero=True, **accuracy_pars)
l, ps_unlensed = pspy_utils.unlensed_ps_from_params(cosmo_params, lmax_sim, raw_cl=True, start_at_zero=True, **accuracy_pars)

ps_lensed_mat   = get_ps_mat(l, ps_lensed)
ps_unlensed_mat = get_unlensed_ps_mat(l, ps_unlensed)

np.save(f"{lensing_dir}/ps_lensed_mat.npy", ps_lensed_mat)
np.save(f"{lensing_dir}/ps_unlensed_mat", ps_unlensed_mat)

fac = l * (l + 1) / (2 * np.pi)
for spec in spectra:
    ps_lensed[spec] *= fac
    ps_unlensed[spec] *= fac
    
so_spectra.write_ps(f"{lensing_dir}/ps_lensed.dat", l, ps_lensed, "Dl", spectra=spectra)
so_spectra.write_ps(f"{lensing_dir}/ps_unlensed.dat", l, ps_unlensed, "Dl", spectra=spectra)
