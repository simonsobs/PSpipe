"""
Compute the connected trispectrum corresponding to the tSZ in the map
"""

from pspy import so_dict, pspy_utils, so_cov, so_map
from pspipe_utils import pspipe_list, log
import numpy as np
import sys, os
import pylab as plt
import time

def get_w2_dict(win_dict, pixsize_map):
    """
    precompute the integral of the square of all different window pairs
    """
    w2_dict = {}
    for c1, ms1 in enumerate(win_dict.keys()):
        for c2, ms2 in enumerate(win_dict.keys()):
            if c1 > c2: continue
            w2_dict[ms1, ms2] = np.sum(win_dict[ms1].data * win_dict[ms2].data * pixsize_map)
            w2_dict[ms2, ms1] = w2_dict[ms1, ms2]

    return w2_dict
    
def tSZ_scaling(nu_ghz):
    TCMB = 2.7255 #Kelvin
    TCMB_uK = 2.7255e6 #micro-Kelvin
    hplanck = 6.626068e-34 #MKS
    kboltz = 1.3806503e-23 #MKS
    nu = 1.e9 * np.asarray(nu_ghz).astype(float)
    X = hplanck * nu / (kboltz * TCMB)
    resp = (X / np.tanh(X / 2.0) - 4.0) * TCMB_uK # into uK_CMB units
    resp /= 10 ** 6 # Boris matrix in in 10^{6} y unit
    return resp




d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_high)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes_for_xar_cov = ["TT", "TE", "ET", "EE"]
else:
    modes_for_xar_cov = spectra

win_dir = d["window_dir"]
cov_dir = "covariances"
plot_dir = "plots/trispectrum_tSZ"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(cov_dir)

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, delimiter="_", return_nu_tag=True)
x_ar_cov_list = pspipe_list.x_ar_cov_order(spec_name_list, nu_tag_list, spectra_order=modes_for_xar_cov)


w_dict = {}
for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
        map_set = f"{sv}_{ar}"
        w_dict[map_set] = so_map.read_map(f"{win_dir}/window_{map_set}_baseline.fits")

first_item  = list(w_dict.keys())[0]
pixsize_map = w_dict[first_item].data.pixsizemap() #all win have the same pixsize, precompute this
w2_dict = get_w2_dict(w_dict, pixsize_map) # precompute the w2 factor instead of doing it in the big loop
        
n_el = len(x_ar_cov_list)
x_ar_non_gaussian_cov_tsz = np.zeros((n_el * n_bins, n_el * n_bins))

trispectrum_yy = np.load("/global/cfs/cdirs/act/data/tlouis/dr6v4/tSZ_trispectrum/reference_trispectrum_planck_level_2_10000_150324.npy")
print(trispectrum_yy.shape)
lmax_mat = trispectrum_yy.shape[0]
l = np.arange(2, lmax_mat + 2)
if type == "Dl": fac = l * (l + 1) / (2 * np.pi)
if type == "Cl": fac = l * 0 + 1
cov_fac = np.outer(fac, fac)

# the effective frequencies have been computed by Benjamin Beringue using equation D4 of: https://arxiv.org/pdf/2007.07289.pdf for the DR6 arrays
tSZ_nu_eff = {}
tSZ_nu_eff["dr6_pa4_f220"] = 226.51
tSZ_nu_eff["dr6_pa5_f090"] = 98.53
tSZ_nu_eff["dr6_pa5_f150"] = 150.92
tSZ_nu_eff["dr6_pa6_f090"] = 97.80
tSZ_nu_eff["dr6_pa6_f150"] = 149.75


for id_el1, x_ar_el1 in enumerate(x_ar_cov_list):
    for id_el2, x_ar_el2 in enumerate(x_ar_cov_list):
        if id_el1 > id_el2: continue

        spec_1, spec_name_1, _ = x_ar_el1
        spec_2, spec_name_2, _ = x_ar_el2
        
        if (spec_1 != "TT") or (spec_2 != "TT"): continue
        
        name_a1, name_b1 = spec_name_1.split("x")
        name_a2, name_b2 = spec_name_2.split("x")
        
        
        nu_a1, nu_b1 = tSZ_nu_eff[name_a1], tSZ_nu_eff[name_b1]
        nu_a2, nu_b2 = tSZ_nu_eff[name_a2], tSZ_nu_eff[name_b2]


        id1_low, id1_high = id_el1 * n_bins, (id_el1 + 1) * n_bins
        id2_low, id2_high = id_el2 * n_bins, (id_el2 + 1) * n_bins
        
        """
        inv_Omega:
        inspired from get_survey_solid_angle of so_windows, we compute the inverse of the solid angle that
        should be relevant for the trispectrum. To convince you that the formula make sense, imagine two map set 1 and 2 with no overlap
        and consider Cov(Cl1_ab, Cl2_ab) inv_omega will be zero in that so the trispectrum contribution Tll'/Omega will be zero
        reflecting the fact that there is indeed no covariance between non overlaping survey.
        """

        inv_Omega = np.sum(w_dict[name_a1].data * w_dict[name_b1].data * w_dict[name_a2].data  * w_dict[name_b2].data * pixsize_map)
        inv_Omega /= (w2_dict[name_a1, name_b1] * w2_dict[name_a2, name_b2])
        
        tSZ_nu_a1, tSZ_nu_b1 = tSZ_scaling(nu_a1), tSZ_scaling(nu_b1)
        tSZ_nu_a2, tSZ_nu_b2 = tSZ_scaling(nu_a2), tSZ_scaling(nu_b2)
        
        nu_scaling = tSZ_nu_a1 * tSZ_nu_b1 * tSZ_nu_a2 * tSZ_nu_b2
        
        trispectrum = trispectrum_yy * nu_scaling

        x_ar_non_gaussian_cov_tsz[id1_low: id1_high, id2_low: id2_high] = so_cov.bin_mat(cov_fac * trispectrum, binning_file, lmax) * inv_Omega
        
        fsky = 1 / (inv_Omega * 4 * np.pi)
        
        log.info(f"({spec_name_1},{spec_name_2}),  ({nu_a1}, {nu_b1}, {nu_a2}, {nu_b2}), fsky = {fsky:.3f}, freq_scaling = {nu_scaling:.3f}")

x_ar_non_gaussian_cov_tsz  = np.triu(x_ar_non_gaussian_cov_tsz) + np.tril(x_ar_non_gaussian_cov_tsz.T, -1)
np.save(f"{cov_dir}/x_ar_non_gaussian_cov_tSZ.npy", x_ar_non_gaussian_cov_tsz)
