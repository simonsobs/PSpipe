"""
Generate window, mcm, and gaussian analytic cov for lensing sim
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

niter = d["niter"]
binned_mcm = d["binned_mcm"]
use_toeplitz_cov = d["use_toeplitz_cov"]
type = d["type"]
binning_file = d["binning_file"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lensing_dir = "lensing"
pspy_utils.create_directory(lensing_dir)

template_car = so_map.read_map(f"{get_data_path()}/binaries/binary_dr6_pa6_f150_downgraded.fits")
template_car = template_car.upgrade(d["upgrade_fac"])
template_car.data = template_car.data.astype("float64")
dist = so_window.get_distance(template_car, rmax=4 * np.pi / 180)
template_car.data[dist.data < 2 ] = 0
window = so_window.create_apodization(template_car, "C1", 2, use_rmax=True) # re-create a dr6 like window
window.plot(file_name=f"{lensing_dir}/window")
window.write_map(f"{lensing_dir}/window.fits")

window_tuple = (window, window)
lmax = int(template_car.get_lmax_limit()) # max l from template pixellisation

mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple,
                                            binning_file,
                                            lmax=lmax,
                                            type=type,
                                            niter=niter,
                                            binned_mcm=binned_mcm,
                                            save_file=f"{lensing_dir}/")

l, ps_lensed = so_spectra.read_ps(f"{lensing_dir}/ps_lensed.dat", spectra=spectra)

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{lensing_dir}/", spin_pairs=spin_pairs)
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
