"""
This script tests the covariance matrix computation with pspy and namaster for spin0 fields.
It is done in HEALPIX pixellisation
"""

import matplotlib
matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils, so_cov
import healpy as hp
import numpy as np
import pylab as plt
import os
import pymaster as nmt
import time


lon, lat = 30, 50
radius = 25
nside = 256
ncomp = 1
lmin= 0
lmax = 3 * nside
nlb = 25
niter = 3
apo_radius_degree_survey = 3
type = "Cl"

test_dir = "result_pspyVSnamaster_cov_spin0"
try:
    os.makedirs(test_dir)
except:
    pass

# Create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=nlb, n_bins=300, file_name="%s/binning.dat" % test_dir)
binning_file="%s/binning.dat" % test_dir

# the templates for the CMB splits
template = so_map.healpix_template(ncomp, nside=nside)
# the templates for the binary mask
binary = so_map.healpix_template(ncomp=1, nside=nside)
# we set pixel inside the disk at 1 and pixel outside at zero
vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
disc = hp.query_disc(nside, vec, radius=radius*np.pi/180)
binary.data[disc] = 1

print("Generate window function")
# we then apodize the survey mask
window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=apo_radius_degree_survey)
window.plot(file_name="%s/window"%(test_dir))


# generate theory power spectrum
ps_theory= {}
import camb

camb_lmin= 0
camb_lmax= 10000
l_camb = np.arange(camb_lmin, camb_lmax)
cosmo_params = {
    "H0": 67.5,
    "As": 1e-10 * np.exp(3.044),
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544
}
pars = camb.set_params(**cosmo_params)
pars.set_for_lmax(camb_lmax, lens_potential_accuracy=1)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

for c,spec in enumerate(["TT", "EE", "BB", "TE"]):
    ps_theory[spec] = powers["total"][:,c][:camb_lmax] * 2 * np.pi/(l_camb * (l_camb + 1))
    ps_theory[spec][0], ps_theory[spec][1] = 0, 0

# Cov mat pspy
print("cov mat pspy")

mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window, binning_file, lmax=lmax, type=type, niter=niter)

survey_id = ["Ta", "Tb", "Tc", "Td"]
survey_name = ["split_0", "split_1", "split_0", "split_1"]

Clth_dict = {}
for name1, id1 in zip(survey_name, survey_id):
    for name2, id2 in zip(survey_name, survey_id):
        spec = id1[0] + id2[0]
        Clth_dict[id1 + id2] = ps_theory[spec][2:lmax+2]

coupling_dict = so_cov.cov_coupling_spin0(window, lmax, niter=niter)
cov_pspy = so_cov.cov_spin0(Clth_dict, coupling_dict, binning_file, lmax, mbb_inv, mbb_inv)


# Cov mat namaster
print("cov mat namaster")

def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    
    return cl_decoupled

cl_tt, cl_ee, cl_bb, cl_te = ps_theory["TT"][0:lmax], ps_theory["EE"][0:lmax], ps_theory["BB"][0:lmax], ps_theory["TE"][0:lmax]

mp_t, mp_q, mp_u = hp.synfast([cl_tt, cl_ee, cl_bb, cl_te],
                              nside,
                              verbose=False)
    
f0, f2 =  nmt.NmtField(window.data, [mp_t]), nmt.NmtField(window.data, [mp_q, mp_u])
b = nmt.NmtBin(nside, nlb=nlb)

lb = b.get_effective_ells()
                              
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(f0, f0, b)
cl_00 = compute_master(f0, f0, w00)
n_ell = len(cl_00[0])
                              
cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0, f0, f0, f0)
covar_00_00 = nmt.gaussian_covariance(cw,
                                      0, 0, 0, 0,
                                      [cl_tt],
                                      [cl_tt],
                                      [cl_tt],
                                      [cl_tt],
                                      w00, wb=w00).reshape([n_ell, 1, n_ell, 1])
                                      
cov_namaster = covar_00_00[:, 0, :, 0]




corr_pspy = so_cov.cov2corr(cov_pspy)
plt.matshow(corr_pspy, vmin=-0.3, vmax=0.3)
plt.savefig("%s/corr_pspsy.png"%(test_dir))
plt.clf()
plt.close()

corr_namaster = so_cov.cov2corr(cov_namaster)
plt.matshow(corr_namaster, vmin=-0.3, vmax=0.3)
plt.savefig("%s/corr_namaster.png"%(test_dir))
plt.clf()
plt.close()


print("cov mat sim")
nsims = 1000
Db_list= []
for iii in range(nsims):
    print (iii)
    cmb = template.copy()
    cmb.data = hp.sphtfunc.synfast(ps_theory["TT"], nside, new=True, verbose=False)
    cmb.data  -= np.mean(cmb.data * window.data)
    alm = sph_tools.get_alms(cmb, window, niter, lmax)
    ls, ps = so_spectra.get_spectra(alm, alm)
    lb, Db = so_spectra.bin_spectra(ls,
                                    ps,
                                    binning_file,
                                    lmax,
                                    type=type,
                                    mbb_inv=mbb_inv)
    Db_list += [Db]

mean = np.mean(Db_list, axis=0)
std = np.std(Db_list, axis=0)

plt.plot(lb, np.sqrt(cov_pspy.diagonal())/np.sqrt(cov_namaster.diagonal()), label="pspy/namaster")
plt.xlabel(r"$\ell$", fontsize=20)
plt.legend()
plt.savefig("%s/compare.png"%(test_dir))
plt.clf()
plt.close()


plt.plot(lb, np.sqrt(cov_namaster.diagonal())/std, label="namaster/sim")
plt.plot(lb, np.sqrt(cov_pspy.diagonal())/std, '.', label="pspy/sim")
plt.xlabel(r"$\ell$", fontsize=20)
plt.legend()
plt.savefig("%s/ratio.png"%(test_dir))
plt.clf()
plt.close()



