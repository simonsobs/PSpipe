"""
Generate gaussian analytic cov for signal only sim
"""
import numpy as np
import pylab as plt
from pspy import pspy_utils, so_map, so_window, so_mcm, so_cov, so_dict, so_spectra
from pspipe_utils import  get_data_path
from pspipe_utils import log
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

test_array = "pa5_f090"
niter = d["niter"]
binned_mcm = d["binned_mcm"]
use_toeplitz_cov = d["use_toeplitz_cov"]
type = d["type"]
binning_file = d["binning_file"]
lmax = d["lmax"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lensing_dir = "lensing"
mcm_dir = "mcms"

pspy_utils.create_directory(lensing_dir)
window = so_map.read_map(d[f"window_T_dr6_{test_array}"])

window_tuple = (window, window)

l, ps_lensed = so_spectra.read_ps(f"{lensing_dir}/ps_lensed.dat", spectra=spectra)

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

log.info("compute mcm and BBl")

mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple,
                                            binning_file,
                                            lmax=lmax,
                                            type=type,
                                            niter=niter,
                                            binned_mcm=binned_mcm,
                                            save_file=f"{lensing_dir}/no_beam")

log.info("done")


Clth_dict = {}
for field1 in ["T", "E", "B"]:
    for id1 in ["a", "b", "c", "d"]:
        for field2 in ["T", "E", "B"]:
            for id2 in ["a", "b", "c", "d"]:
                name = field1 + id1 + field2 + id2
                Clth_dict[name] = ps_lensed[field1 + field2][2:lmax] # no noise here
                
if use_toeplitz_cov == True:
    log.info("we will use the toeplitz approximation")
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

log.info("compute gaussian analytic cov")
coupling_dict = so_cov.cov_coupling_spin0and2_simple(window,
                                                    lmax,
                                                    niter=niter,
                                                    l_exact=l_exact,
                                                    l_band=l_band,
                                                    l_toep=l_toep)
                                                    
analytic_cov = so_cov.cov_spin0and2(Clth_dict,
                                    coupling_dict,
                                    binning_file,
                                    lmax,
                                    mbb_inv,
                                    mbb_inv,
                                    cov_T_E_only=False,
                                    binned_mcm=binned_mcm)

np.save(f"{lensing_dir}/analytic_cov.npy", analytic_cov)
