"""
This script tests the spectra computation with pspy and namaster for spin0 fields.
It is done in HEALPIX pixellisation
"""
import matplotlib
matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils
import healpy as hp
import numpy as np
import pylab as plt
import os
import pymaster as nmt
import time

# We  specify the HEALPIX survey parameter, it will be a disk of radius 25 degree centered on longitude 30 degree and latitude 50 degree
# It will have a resolution nside=512
lon, lat = 30, 50
radius = 25
nside = 512
# ncomp=1 mean that we are going to use only spin0 field (ncomp=3 for spin 0 and 2 fields
ncomp = 1
# clfile are the camb lensed power spectra
clfile = "bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
# the maximum multipole to consider
lmax = 3 * nside - 1
# the number of iteration in map2alm
niter = 3
# the noise on the spin0 component
rms_uKarcmin_T = 20
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey = 1
# the number of holes in the point source mask
source_mask_nholes = 100
# the radius of the holes (in arcminutes)
source_mask_radius = 10
# the apodisation lengh for the point source mask (in degree)
apo_radius_degree_mask = 0.3
# the type of power spectrum (Cl or Dl)
type = "Cl"

test_dir = "result_pspyVSnamaster_spin0"
try:
    os.makedirs(test_dir)
except:
    pass

# Create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=40, n_bins=300, file_name="%s/binning.dat" % test_dir)
binning_file="%s/binning.dat" % test_dir

# the templates for the CMB splits
template = so_map.healpix_template(ncomp, nside=nside)
# the templates for the binary mask
binary = so_map.healpix_template(ncomp=1, nside=nside)
# we set pixel inside the disk at 1 and pixel outside at zero
vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
disc = hp.query_disc(nside, vec, radius=radius*np.pi/180)
binary.data[disc] = 1


print("Generate noisy CMB realisation")
# First let's generate a CMB realisation
cmb = template.synfast(clfile)
split = cmb.copy()
# let's add noise to it with rms 20 uk.arcmin
noise = so_map.white_noise(split, rms_uKarcmin_T=rms_uKarcmin_T)
split.data += noise.data

split.plot(file_name="%s/noisy_cmb" % (test_dir))


print("Generate window function")
# we then apodize the survey mask
window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=apo_radius_degree_survey)
# we create a point source mask
mask = so_map.simulate_source_mask(binary, n_holes=source_mask_nholes, hole_radius_arcmin=source_mask_radius)
# ... and we apodize it
mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=apo_radius_degree_mask)
# the window is given by the product of the survey window and the mask window
window.data *= mask.data

# let's look at it
window.plot(file_name="%s/window" % (test_dir), hp_gnomv=(lon, lat, 3500, 1))


print("Compute spin0 power spectra a la pspy")
t0=time.time()
# Compute spin 0 spectra a la pspy
mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window, binning_file, lmax=lmax, type=type, niter=niter)
alm = sph_tools.get_alms(split, window, niter, lmax)
l, ps = so_spectra.get_spectra(alm)
lb, Cb_pspy = so_spectra.bin_spectra(l, ps, binning_file, lmax, type=type, mbb_inv=mbb_inv)
print("pspy run in %.2f s"%(time.time()-t0))

# Compute spin 0 spectra a la namaster
print("Compute spin0 power spectra a la namaster")
nlb = 40
field = nmt.NmtField(window.data, [split.data])
cl_coupled = nmt.compute_coupled_cell(field, field)
b = nmt.NmtBin(nside, nlb=nlb)
lb = b.get_effective_ells()
w0 = nmt.NmtWorkspace()
w0.compute_coupling_matrix(field, field, b)
Cb_namaster = w0.decouple_cell(cl_coupled)
print("namaster run in %.2f s"%(time.time()-t0))


# Plot the spectra
plt.plot(lb, Cb_pspy*lb**2/(2*np.pi), label="pspy")
plt.plot(lb, Cb_namaster[0]*lb**2/(2*np.pi), '.', label="namaster")
plt.ylabel(r"$D_{\ell}$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.legend()
plt.savefig("%s/spectra.png"%(test_dir))
plt.clf()
plt.close()

# Plot the fractionnal difference
plt.plot(lb, (Cb_pspy-Cb_namaster[0])/Cb_pspy)
plt.ylabel(r"$(D^{\rm pspy}_{\ell}-D^{\rm namaster}_{\ell})/D^{\rm pspy}_{\ell}$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.savefig("%s/fractional.png"%(test_dir))
plt.clf()
plt.close()

