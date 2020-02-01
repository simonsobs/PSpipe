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

#First let's generate a CMB realisation
cmb = template.synfast(clfile)
split = cmb.copy()
#let's add noise to it with rms 20 uk.arcmin in T ans sqrt(2)xthat in pol
noise = so_map.white_noise(split, rms_uKarcmin_T=rms_uKarcmin_T)
split.data += noise.data

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

mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window, binning_file, lmax=lmax, type=type, niter=niter)
alms = sph_tools.get_alms(split, window, niter, lmax)
l, ps = so_spectra.get_spectra(alms, spectra=spectra)
lb_py, Cb_pspy = so_spectra.bin_spectra(l, ps, binning_file, lmax, type=type, mbb_inv=mbb_inv, spectra=spectra)

# Compute spin 0 spin 2 spectra a la namaster

nlb = 40

field_0 = nmt.NmtField(window[0].data, [split.data[0]])
field_2 = nmt.NmtField(window[1].data, [split.data[1],split.data[2]])

b = nmt.NmtBin(nside, nlb=nlb)
lb = b.get_effective_ells()

w0 = nmt.NmtWorkspace()
w0.compute_coupling_matrix(field_0, field_0, b)
w1 = nmt.NmtWorkspace()
w1.compute_coupling_matrix(field_0, field_2, b)
w2 = nmt.NmtWorkspace()
w2.compute_coupling_matrix(field_2, field_2, b)

def compute_master(f_a, f_b, wsp) :
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

Cb_namaster={}
Cb_namaster["TT"] = compute_master(field_0, field_0, w0)[0]
spin1 = compute_master(field_0, field_2, w1)
Cb_namaster["TE"] = spin1[0]
Cb_namaster["TB"] = spin1[1]
Cb_namaster["ET"] = Cb_namaster["TE"]
Cb_namaster["BT"] = Cb_namaster["TB"]
spin2 = compute_master(field_2, field_2, w2)
Cb_namaster["EE"] = spin2[0]
Cb_namaster["EB"] = spin2[1]
Cb_namaster["BE"] = spin2[2]
Cb_namaster["BB"] = spin2[3]


plt.figure(figsize=(20, 15))
for c,f in enumerate(spectra):
    plt.subplot(3, 3, c+1)
    plt.plot(lb, Cb_namaster[f]*lb**2/(2*np.pi), label="namaster")
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
    plt.plot(lb, (Cb_namaster[f]-Cb_pspy[f])/Cb_pspy[f])
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.ylabel(r"$\Delta D^{%s}_{\ell}/D^{%s}_{\ell}$"%(f,f), fontsize=20)

    if c == 0:
        plt.legend()
plt.savefig("%s/fractional.png"%(test_dir))
plt.clf()
plt.close()





