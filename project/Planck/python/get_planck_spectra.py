'''
This script is used to compute the power spectra of the Planck data.
To run it:
python get_planck_spectra.py global.dict
'''

import numpy as np
import healpy as hp
from pspy import so_dict, so_map, so_mcm, sph_tools, so_spectra, pspy_utils
import sys
import time
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcms_dir = "mcms"
spectra_dir = "spectra"

pspy_utils.create_directory(spectra_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE","BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

freqs = d["freqs"]
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
remove_mono_dipo_t = d["remove_mono_dipo_T"]
remove_mono_dipo_pol = d["remove_mono_dipo_pol"]
splits = d["splits"]

experiment = "Planck"

alms = {}

print("Compute Planck 2018 spectra")

for freq in freqs:
    maps = d["map_%s" % freq]
    for hm, map in zip(splits, maps):
        
        window_t = so_map.read_map("%s/window_T_%s_%s-%s.fits" % (windows_dir, experiment, freq, hm))
        window_pol = so_map.read_map("%s/window_P_%s_%s-%s.fits" % (windows_dir, experiment, freq, hm))
        window_tuple = (window_t, window_pol)
        del window_t, window_pol

        pl_map = so_map.read_map("%s" % map, fields_healpix=(0, 1, 2))
        pl_map.data *= 10**6
        cov_map = so_map.read_map("%s" % map, fields_healpix=4)
        badpix = (cov_map.data == hp.pixelfunc.UNSEEN)
        for i in range(3):
            pl_map.data[i][badpix] = 0.0

        if remove_mono_dipo_t:
            pl_map.data[0] = planck_utils.subtract_mono_di(pl_map.data[0], window_tuple[0].data, pl_map.nside)
        if remove_mono_dipo_pol:
            pl_map.data[1] = planck_utils.subtract_mono_di(pl_map.data[1], window_tuple[1].data, pl_map.nside)
            pl_map.data[2] = planck_utils.subtract_mono_di(pl_map.data[2], window_tuple[1].data, pl_map.nside)

        alms[hm, freq] = sph_tools.get_alms(pl_map, window_tuple, niter, lmax)


for c1, freq1 in enumerate(freqs):
    for c2, freq2 in enumerate(freqs):
        if c1 > c2: continue
        for s1, hm1 in enumerate(splits):
            for s2, hm2 in enumerate(splits):
                if (s1 > s2) & (c1 == c2): continue
                
                prefix= "%s/%s_%sx%s_%s-%sx%s" % (mcms_dir, experiment, freq1, experiment, freq2, hm1, hm2)

                mcm_inv, mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix, spin_pairs=spin_pairs, unbin=True)

                l, ps = so_spectra.get_spectra(alms[hm1,freq1], alms[hm2,freq2], spectra=spectra)
                spec_name = "%s_%sx%s_%s-%sx%s" % (experiment, freq1, experiment, freq2, hm1, hm2)
                l, cl, lb, Db = planck_utils.process_planck_spectra(l,
                                                                    ps,
                                                                    binning_file,
                                                                    lmax,
                                                                    mcm_inv=mcm_inv,
                                                                    spectra=spectra)

                so_spectra.write_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name), lb ,Db, type=type, spectra=spectra)
                so_spectra.write_ps("%s/spectra_unbin_%s.dat" % (spectra_dir, spec_name), l, cl, type=type, spectra=spectra)


