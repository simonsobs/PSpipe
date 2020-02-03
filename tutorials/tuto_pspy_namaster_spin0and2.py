"""
This is a test of spectra generation with pspy and namaster for spin0 and 2 fields.
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
# the noise on the spin0 component, if not specified, the noise in polarisation wil be sqrt(2)x that
rms_uKarcmin_T = 20
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey = 1
# the number of holes in the point source mask
source_mask_nholes = 10
# the radius of the holes (in arcminutes)
source_mask_radius = 30
# the apodisation lengh for the point source mask (in degree)
apo_radius_degree_mask = 1
# the type of power spectrum (Cl or Dl)
type = "Cl"

test_dir = "result_pspyVSnamaster_spin0and2"
try:
    os.makedirs(test_dir)
except:
    pass

# create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=40, n_bins=300, file_name="%s/binning.dat"%test_dir)
binning_file="%s/binning.dat" % test_dir



# the templates for the CMB splits
template = so_map.healpix_template(ncomp, nside=nside)
# the templates for the binary mask
binary = so_map.healpix_template(ncomp=1, nside=nside)
# we set pixel inside the disk at 1 and pixel outside at zer
vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
disc = hp.query_disc(nside, vec, radius=radius*np.pi/180)
binary.data[disc] = 1

print("Generate CMB realisation")

#First let's generate a CMB realisation
cmb = template.synfast(clfile)
split = cmb.copy()
#let's add noise to it with rms 20 uk.arcmin in T ans sqrt(2)xthat in pol
noise = so_map.white_noise(split, rms_uKarcmin_T=rms_uKarcmin_T)
split.data += noise.data

split.plot(file_name="%s/noisy_cmb" % (test_dir))


print("Generate window function")

#we then apodize the survey mask
window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=apo_radius_degree_survey)
#we create a point source mask
mask = so_map.simulate_source_mask(binary, n_holes=source_mask_nholes, hole_radius_arcmin=source_mask_radius)
#... and we apodize it
mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=apo_radius_degree_mask)
#the window is given by the product of the survey window and the mask window
window.data *= mask.data

#let's look at it
window.plot(file_name="%s/window"%(test_dir), hp_gnomv=(lon,lat,3500,1))

#for spin0 and 2 the window need to be a tuple made of two objects
#the window used for spin0 and the one used for spin 2
window = (window,window)



# Compute spin 0 spin 2 spectra a la pspy
print("Compute spin0 and spin2 power spectra a la pspy")
t0=time.time()
mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window, binning_file, lmax=lmax, type=type, niter=niter)
alms = sph_tools.get_alms(split, window, niter, lmax)
l, ps = so_spectra.get_spectra(alms, spectra=spectra)
lb_py, Cb_pspy = so_spectra.bin_spectra(l, ps, binning_file, lmax, type=type, mbb_inv=mbb_inv, spectra=spectra)
print("pspy run in %.2f s"%(time.time()-t0))


print("Compute spin0 and spin2 power spectra a la namaster")
# Compute spin 0 spin 2 spectra a la namaster
t0=time.time()

field_0 = nmt.NmtField(window[0].data, [split.data[0]])
field_2 = nmt.NmtField(window[1].data, [split.data[1], split.data[2]])

nlb = 40
b = nmt.NmtBin(nside, nlb=nlb)
lb_namaster = b.get_effective_ells()

wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(field_0, field_2, b, is_teb=True,
                            n_iter=niter, lmax_mask=lmax)

# Compute mode-coupled Cls (for each pair of fields)
cl_coupled_00 = nmt.compute_coupled_cell(field_0, field_0)
cl_coupled_02 = nmt.compute_coupled_cell(field_0, field_2)
cl_coupled_22 = nmt.compute_coupled_cell(field_2, field_2)

# Bundle them up
cls_coupled = np.array([cl_coupled_00[0],  # TT
                        cl_coupled_02[0],  # TE
                        cl_coupled_02[1],  # TB
                        cl_coupled_22[0],  # EE
                        cl_coupled_22[1],  # EB
                        cl_coupled_22[2],  # BE
                        cl_coupled_22[3]])  # BB

# Invert MCM
cls_uncoupled = wsp.decouple_cell(cls_coupled)

Clb_namaster = {}
Clb_namaster["TT"] = cls_uncoupled[0]
Clb_namaster["TE"] = cls_uncoupled[1]
Clb_namaster["TB"] = cls_uncoupled[2]
Clb_namaster["ET"] = Clb_namaster["TE"]
Clb_namaster["BT"] = Clb_namaster["TB"]
Clb_namaster["EE"] = cls_uncoupled[3]
Clb_namaster["EB"] = cls_uncoupled[4]
Clb_namaster["BE"] = cls_uncoupled[5]
Clb_namaster["BB"] = cls_uncoupled[6]

print("namaster run in %.2f s"%(time.time()-t0))

plt.figure(figsize=(20, 15))
for c,f in enumerate(spectra):
    plt.subplot(3, 3, c+1)
    plt.plot(lb_namaster, Clb_namaster[f]*lb_namaster**2/(2*np.pi), label="namaster")
    plt.plot(lb_py, Cb_pspy[f]*lb_py**2/(2*np.pi), '.', label="pspy")
    plt.ylabel(r"$D^{%s}_{\ell}$"%f, fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    if c == 0:
        plt.legend()
plt.savefig("%s/spectra.png"%(test_dir))
plt.clf()
plt.close()

plt.figure(figsize=(20, 15))
for c,f in enumerate(spectra):
    plt.subplot(3, 3, c+1)
    plt.plot(lb_namaster, (Clb_namaster[f]-Cb_pspy[f])/Cb_pspy[f])
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.ylabel(r"$\Delta D^{%s}_{\ell}/D^{%s}_{\ell}$"%(f,f), fontsize=20)

    if c == 0:
        plt.legend()
plt.savefig("%s/fractional.png"%(test_dir))
plt.clf()
plt.close()





