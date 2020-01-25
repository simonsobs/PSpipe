'''
This script is used to generate simulations of Planck data and compute their power spectra.
to run it:
python planck_sim_spectra.py global.dict
Note that we are using homogeneous non white noise here
'''
import numpy as np
import healpy as hp
from pspy import so_dict, so_map, so_mcm, sph_tools, so_spectra, pspy_utils, so_mpi
import sys
from pixell import curvedsky, powspec
import time
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcms_dir = "mcms"
noise_model_dir = "noise_model"
bestfit_dir = "best_fits"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

iStart = d["iStart"]
iStop = d["iStop"]
freqs = d["freqs"]
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
use_ffp10 = d["use_ffp10"]
binning_file = d["binning_file"]
remove_mono_dipo_t = d["remove_mono_dipo_T"]
remove_mono_dipo_pol = d["remove_mono_dipo_pol"]
splits = d["splits"]
nsplits = len(splits)

sims_dir = "sim_spectra"
if use_ffp10:
    sims_dir = "sim_spectra_ffp10"

pspy_utils.create_directory(sims_dir)

exp = "Planck"

nside = 2048
ncomp = 3
template = so_map.healpix_template(ncomp, nside)
pixwin=hp.pixwin(nside)

bestfit=np.load("%s/bestfit_matrix.npy" % bestfit_dir)
l, nl_array_t, nl_array_pol = planck_utils.noise_matrix(noise_model_dir, exp, freqs, lmax, nsplits)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:

    t0=time.time()
    
    alms={}

    sim_alm = curvedsky.rand_alm(bestfit, lmax=lmax)
    
    if use_ffp10 == False:
        nlms = planck_utils.generate_noise_alms(nl_array_t, nl_array_pol, lmax, nsplits)

    for f_id, freq in enumerate(freqs):
        
        maps = d["map_%s" % freq] # In use for the missing pixels
        
        freq_alm = np.zeros((3, sim_alm.shape[1]), dtype="complex")
        freq_alm[0] = sim_alm[0 + f_id * 3].copy()
        freq_alm[1] = sim_alm[1 + f_id * 3].copy()
        freq_alm[2] = sim_alm[2 + f_id * 3].copy()

        for hm, map, k in zip(splits, maps, np.arange(nsplits)):
            
            hm_alms = freq_alm.copy()
            
            l, bl_T = np.loadtxt(d["beam_%s_%s_T" % (freq, hm)], unpack=True)
            l, bl_pol = np.loadtxt(d["beam_%s_%s_pol" % (freq, hm)], unpack=True)

            hm_alms[0] = hp.sphtfunc.almxfl(hm_alms[0], bl_T)
            hm_alms[1] = hp.sphtfunc.almxfl(hm_alms[1], bl_pol)
            hm_alms[2] = hp.sphtfunc.almxfl(hm_alms[2], bl_pol)

            if use_ffp10 == False:
                hm_alms[0] +=  nlms["T", k][f_id]
                hm_alms[1] +=  nlms["E", k][f_id]
                hm_alms[2] +=  nlms["B", k][f_id]
            
            for i in range(3):
                hm_alms[i] = hp.sphtfunc.almxfl(hm_alms[i], pixwin)

            pl_map = sph_tools.alm2map(hm_alms, template)
            if use_ffp10 == True:
                noise_map = so_map.read_map("%s/%s/ffp10_noise_%s_%s_map_mc_%05d.fits"%(d["ffp10_dir"], freq, freq, hm, iii))
                noise_map.data *= 10**6
                pl_map.data += noise_map.data

            window_t = so_map.read_map("%s/window_T_%s_%s-%s.fits" % (windows_dir ,exp, freq, hm))
            window_pol = so_map.read_map("%s/window_P_%s_%s-%s.fits" % (windows_dir, exp, freq, hm))
            window_tuple = (window_t, window_pol)
            del window_t, window_pol

            cov_map = so_map.read_map("%s" % map, fields_healpix=4)
            badpix = (cov_map.data == hp.pixelfunc.UNSEEN)
            for i in range(3):
                pl_map.data[i][badpix] = 0.0
            if remove_mono_dipo_t:
                pl_map.data[0] = planck_utils.subtract_mono_di(pl_map.data[0], window_tuple[0].data, pl_map.nside )
            if remove_mono_dipo_pol:
                pl_map.data[1] = planck_utils.subtract_mono_di(pl_map.data[1], window_tuple[1].data, pl_map.nside)
                pl_map.data[2] = planck_utils.subtract_mono_di(pl_map.data[2], window_tuple[1].data, pl_map.nside)

            alms[hm, freq] = sph_tools.get_alms(pl_map, window_tuple, niter, lmax)

    Db_dict = {}
    spec_name_list = []
    for c1, freq1 in enumerate(freqs):
        for c2, freq2 in enumerate(freqs):
            if c1 > c2: continue
            for s1, hm1 in enumerate(splits):
                for s2, hm2 in enumerate(splits):
                    if (s1 > s2) & (c1 == c2): continue
                
                    prefix= "%s/%s_%sx%s_%s-%sx%s" % (mcms_dir, exp, freq1, exp, freq2, hm1, hm2)

                    mcm_inv, mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix, spin_pairs=spin_pairs, unbin=True)

                    l, ps = so_spectra.get_spectra(alms[hm1, freq1], alms[hm2, freq2], spectra=spectra)
                    spec_name = "%s_%sx%s_%s-%sx%s" % (exp, freq1, exp, freq2, hm1, hm2)
                    l, cl, lb, Db = planck_utils.process_planck_spectra(l,
                                                                        ps,
                                                                        binning_file,
                                                                        lmax,
                                                                        mcm_inv=mcm_inv,
                                                                        spectra=spectra)
                    spec_name_list += [spec_name]
                    so_spectra.write_ps("%s/sim_spectra_%s_%04d.dat"%(sims_dir, spec_name, iii), lb, Db, type=type, spectra=spectra)
                    so_spectra.write_ps("%s/sim_spectra_unbin_%s_%04d.dat"%(sims_dir,spec_name,iii), l, cl, type=type, spectra=spectra)

    print("sim %04d take %.02f seconds to compute" % (iii, time.time()-t0))
