"""
Compare full sky power spectrum with cut sky power spectrum and CAMB prediction
"""
import sys
import pickle

import pylab as plt
import numpy as np
from pspipe_utils import pspipe_list, log, best_fits
from pspy import so_dict, so_spectra, pspy_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

cosmo_params = d["cosmo_params"]
type = d["type"]
lmax = d["lmax"]
binning_file = "data/BIN_ACTPOL_50_4_SC_large_bin_at_low_ell"

ps_full_sky = {}
ps_full_sky["TT"], ps_full_sky["EE"], ps_full_sky["BB"], ps_full_sky["TE"], ps_full_sky["EB"], ps_full_sky["TB"] = pickle.load(open("data/cmb_seed1_fullsky_nobeam.p", 'rb'))
ells = np.arange(10001)
facs = ells * (ells +  1) / (2 * np.pi)

l_th, ps_th = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500, **d["accuracy_params"])

for spectrum in ["TT", "EE"]:
    
    lb_th, psb_th = pspy_utils.naive_binning(l_th, ps_th[spectrum], binning_file, lmax)
    lb, psb_fsky = pspy_utils.naive_binning(ells, ps_full_sky[spectrum] * facs, binning_file, lmax)


    l, ps_std = so_spectra.read_ps(f"spectra_components_std/Dl_dr6_pa6_f150xdr6_pa6_f150_cmb_seed1xcmb_seed1.dat", spectra=spectra)
    l, ps_no_kspace = so_spectra.read_ps(f"spectra_components_no_kspace/Dl_dr6_pa6_f150xdr6_pa6_f150_cmb_seed1xcmb_seed1.dat", spectra=spectra)
    
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.semilogy()
    plt.plot(lb, psb_fsky, ".", label="full sky")
    plt.plot(l, ps_std[spectrum], label="cut sky")
    plt.plot(l, ps_no_kspace[spectrum], "--", label="cut sky no kspace")
    plt.plot(lb_th, psb_th, label="CAMB")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(lb, psb_fsky/psb_fsky, label="full sky")
    plt.plot(l, ps_std[spectrum]/psb_fsky, label="cut sky")
    plt.plot(l, ps_no_kspace[spectrum]/psb_fsky, label="cut sky no kspace")
    plt.plot(lb_th, psb_th/psb_fsky, label="CAMB")
    plt.show()

