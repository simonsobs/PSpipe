import matplotlib
matplotlib.use('Agg')
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils
import os
import pymaster as nmt
from copy import deepcopy
import time

#We specify the HEALPIX survey parameter, it will be a disk of radius 25 degree centered on longitude 30 degree and latitude 50 degree
# It will have a resolution nside=512
lon, lat = 30, 50
radius = 25
nside = 512
# ncomp=3 mean that we are going to use spin0 and 2 field
ncomp = 3
# specify the order of the spectra, this will be the order used in pspy
# note that if you are doing cross correlation between galaxy and kappa for example, you should follow a similar structure
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
# clfile are the camb lensed power spectra
clfile = "bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
# the maximum multipole to consider
lmax = 3 * nside - 1
# the number of iteration in map2alm
niter = 3
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey = 5
# spectra type
type = "Cl"
# the templates for the CMB splits
template = so_map.healpix_template(ncomp,nside=nside)

#  we set pixel inside the disk at 1 and pixel outside at zero
binary = so_map.healpix_template(ncomp=1, nside=nside)
vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
disc = hp.query_disc(nside, vec, radius=radius*np.pi/180)
binary.data[disc] = 1

test_dir = "result_pspyVSnamaster_pureBB"
try:
    os.makedirs(test_dir)
except:
    pass

# create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=50, n_bins=300, file_name="%s/binning.dat" % test_dir)
binning_file = "%s/binning.dat" % test_dir

window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=apo_radius_degree_survey)
    
cmb=template.synfast(clfile)

# Compute spin 0 spin 2 spectra a la pspy
t0=time.time()

mbb_inv_pure, Bbl_pure = so_mcm.mcm_and_bbl_spin0and2((window,window),
                                                           binning_file,
                                                           lmax=lmax,
                                                           niter=niter,
                                                           type=type,
                                                           pure=True)

alm_pure = sph_tools.get_pure_alms(cmb, (window,window), niter, lmax)
l, ps_pure = so_spectra.get_spectra(alm_pure, alm_pure, spectra=spectra)

lb, ps_dict_pure = so_spectra.bin_spectra(l,
                                          ps_pure,
                                          binning_file,
                                          lmax,
                                          type=type,
                                          mbb_inv=mbb_inv_pure,
                                          spectra=spectra)

print("pspy run in %.2f s"%(time.time()-t0))

# Compute pure spin 2 spectra a la namaster
t0=time.time()

def compute_master(f_a,f_b,wsp) :
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

fyp = nmt.NmtField(window.data,[cmb.data[1], cmb.data[2]], purify_e=True, purify_b=True, n_iter_mask_purify=niter, n_iter=niter)
b = nmt.NmtBin(nside, nlb=50)
w_yp = nmt.NmtWorkspace(); w_yp.compute_coupling_matrix(fyp, fyp, b, n_iter=niter)
leff = b.get_effective_ells()
data = compute_master(fyp, fyp, w_yp)

print("namaster run in %.2f s"%(time.time()-t0))

plt.plot(leff, data[3])
plt.plot(lb, ps_dict_pure["BB"], 'o')
plt.savefig("%s/BB.png"%test_dir)
plt.clf()
plt.close()
