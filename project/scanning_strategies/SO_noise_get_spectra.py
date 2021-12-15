"""
This script compute the auto and cross power spectra for the different scanning strategy
"""

import healpy as hp
import pylab as plt
import numpy as np
import sys
from pspy import so_map, so_window, sph_tools, so_mcm, pspy_utils, so_spectra, so_dict
import SO_noise_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

scan_list = d["scan_list"]
lmax = d["lmax"]
apo_type = d["apo_type"]
apo_radius_degree = d["apo_radius_degree"]
niter = d["niter"]
spectra = d["spectra"]
split_list = d["split_list"]
runs = d["runs"]
clfile = d["clfile"]
binning_file = d["binning_file_name"]
bin_size = d["bin_size"]
map_plot_range = d["map_plot_range"]
vrange = 3 * [map_plot_range]
K_to_muK = 10**6

include_cmb = True

lth, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)

pspy_utils.create_binning_file(bin_size=bin_size, n_bins=300, file_name=binning_file )

map_dir = d["map_dir"]
mcm_dir = "mcms"
spectra_dir = "spectra"
plot_dir = "plot/map"
window_dir = "windows"

pspy_utils.create_directory(window_dir)
pspy_utils.create_directory(mcm_dir)
pspy_utils.create_directory(spectra_dir)
pspy_utils.create_directory(plot_dir)

template = so_map.healpix_template(ncomp=3, nside=1024)
cmb = template.synfast(clfile)

Db_dict = {}
for scan in scan_list:

    print("processing %s" % scan)
    binary = so_map.healpix_template(ncomp=1, nside=1024)
    binary.data[:] = 1
    
    sim = {}
    for split in split_list:
        noise_map = so_map.read_map("%s/%s_telescope_all_time_all_map.fits" % (map_dir[split], scan))
        binary.data[noise_map.data[0] == hp.pixelfunc.UNSEEN] = 0
        # the sqrt(2) is because each split has noise corresponding to the whole survey
        noise_map.data[:] *= K_to_muK * np.sqrt(2)
        
        noise_map.plot(file_name="%s/%s_map_%s" % (plot_dir, split, scan), color_range=vrange)
        
        if include_cmb == True:
            sim[split] = cmb.copy()
            sim[split].data[:] += noise_map.data[:]
        else:
            sim[split] = noise_map.copy()
            
    binary.plot(file_name="%s/binary_%s" % (plot_dir, scan))

    for run in runs:

        window = so_window.create_apodization(binary,
                                              apo_type=apo_type,
                                              apo_radius_degree=apo_radius_degree)
    
        if run == "weighted":
            print("Use hits maps")
            hmap = so_map.read_map("%s/%s_telescope_all_time_all_hmap.fits" % (map_dir["split0"], scan))
            window.data *= hmap.data
        else:
            print("Uniform weighting")
        
                
        window.write_map("%s/window_%s_%s.fits" % (window_dir, scan, run))

        window.plot(file_name="%s/window_%s_%s" % (plot_dir, scan, run), title=scan)
    
        window = (window, window)
    
        mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window,
                                                    binning_file,
                                                    lmax=lmax,
                                                    type="Dl",
                                                    niter=niter,
                                                    save_file="%s/%s_%s"%(mcm_dir, scan, run))
    
        alm = {}
        for split in split_list:
            alm[split] = sph_tools.get_alms(sim[split], window, niter, lmax)
            
        for c0, s0 in enumerate(split_list):
            for c1, s1 in enumerate(split_list):
                if c1 > c0: continue

                spec_name = "%s_%sx%s_%s" % (scan, s0, s1, run)

                l, ps = so_spectra.get_spectra(alm[s0], alm[s1], spectra=spectra)
                lb, Db_dict[spec_name] = so_spectra.bin_spectra(l,
                                                                ps,
                                                                binning_file,
                                                                lmax,
                                                                type="Dl",
                                                                mbb_inv=mbb_inv,
                                                                spectra=spectra)

                so_spectra.write_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name),
                                    lb,
                                    Db_dict[spec_name],
                                    type="Dl",
                                    spectra=spectra)

        
