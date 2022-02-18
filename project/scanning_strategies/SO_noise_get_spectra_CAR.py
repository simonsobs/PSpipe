"""
This script compute the auto and cross power spectra for the different scanning strategy
"""

def downgraded_plot(map, file_name, color_range=None, down_factor=4):
    map_low_res = map.copy()
    map_low_res = map_low_res.downgrade(down_factor)
    map_low_res.plot(file_name=file_name, color_range=color_range)
    
import healpy as hp
import pylab as plt
import numpy as np
import sys
from pspy import so_map, so_window, sph_tools, so_mcm, pspy_utils, so_spectra, so_dict, so_mpi
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
color_range = d["map_plot_range"]
skip_from_edges = d["skip_from_edges"]
cross_linking_threshold = d["cross_linking_threshold"]
K_to_muK = 10**6
include_cmb = d["include_CMB"]

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

template = so_map.read_map("%s/lat01_s25_small_tiles_fullfp_f150_1pass_2way_set00_map.fits" % (map_dir))
cmb = template.synfast(clfile)

n_scans = len(scan_list)
print("number of scan to compute : %s" % n_scans)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_scans - 1)

for task in subtasks:
    task = int(task)
    scan = scan_list[task]

    binary = so_map.read_map("%s/lat01_s25_%s_fullfp_f150_1pass_2way_set00_div.fits" % (map_dir, scan))
    binary.data[:] = 1
    
    sim = {}
    for count, split in enumerate(split_list):
        noise_map = so_map.read_map("%s/lat01_s25_%s_fullfp_f150_1pass_2way_set%02d_map.fits" % (map_dir, scan, count))
        binary.data[noise_map.data[0] == 0] = 0
        noise_map.data[:] *= K_to_muK # should it be * np.sqrt(2) ?
        
        downgraded_plot(noise_map, "%s/%s_map_%s" % (plot_dir, split, scan), color_range=color_range)
        
        if include_cmb == True:
            sim[split] = cmb.copy()
            sim[split].data[:] += noise_map.data[:]
        else:
            sim[split] = noise_map.copy()
    
        if (cross_linking_threshold != 1) & (count == 0):
            print("mask region with poor cross linking")
            
            xlink_car = binary.copy()

            x_link_healpix = so_map.read_map("%s/lat01_s25_%s_fullfp_f150_1pass_2way_set%02d_crosslinking.fits" % (map_dir.replace("car", "healpix"), scan, count))
            x_link_I, x_link_Q, x_link_U = x_link_healpix.data
            
            temp_healpix = so_map.healpix_template(1, x_link_healpix.nside, x_link_healpix.coordinate)

            temp_healpix.data = np.sqrt(x_link_Q ** 2 + x_link_U ** 2) / x_link_I
            id = np.where(x_link_I == 0)
            temp_healpix.data[id] = 0
            
            
            xlink_car = so_map.healpix2car(temp_healpix, xlink_car)
            downgraded_plot(xlink_car, "%s/cross_link_%s" % (plot_dir, scan))

            binary.data[xlink_car.data > cross_linking_threshold] = 0
            
    
    downgraded_plot(binary, "%s/binary_%s" % (plot_dir, scan))

    if skip_from_edges != 0:
    
        binary.data[0,:] = 0
        binary.data[-1,:] = 0

        dist = so_window.get_distance(binary, rmax=(2 * skip_from_edges) * np.pi / 180)

        binary.data[dist.data < skip_from_edges] = 0
        downgraded_plot(binary, "%s/binary_skip_%s" % (plot_dir, scan))

    for run in runs:

        window = so_window.create_apodization(binary,
                                              apo_type=apo_type,
                                              apo_radius_degree=apo_radius_degree)
    
        if run == "weighted":
            print("Use hits maps")
            hmap = so_map.read_map("%s/lat01_s25_%s_fullfp_f150_1pass_2way_set00_div.fits" % (map_dir, scan))
            window.data *= hmap.data
        else:
            print("Uniform weighting")
        
                
        window.write_map("%s/window_%s_%s.fits" % (window_dir, scan, run))
        
        downgraded_plot(window, "%s/window_%s_%s" % (plot_dir, scan, run))


    
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
                lb, Db_dict = so_spectra.bin_spectra(l,
                                                     ps,
                                                     binning_file,
                                                     lmax,
                                                     type="Dl",
                                                     mbb_inv=mbb_inv,
                                                     spectra=spectra)

                so_spectra.write_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name),
                                    lb,
                                    Db_dict,
                                    type="Dl",
                                    spectra=spectra)

        
