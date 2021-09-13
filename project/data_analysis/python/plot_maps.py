"""
This script plot all maps before and after filtering
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra
from pixell import enmap
import numpy as np
import healpy as hp
import sys
import data_analysis_utils
import time


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
deconvolve_pixwin = d["deconvolve_pixwin"]

window_dir = "windows"
mcm_dir = "mcms"
specDir = "spectra"
plot_dir = "plots/maps/"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(specDir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

ncomp = 3

nsplit = {}
pixwin_l = {}
color_range = [500, 150, 150]

for sv in surveys:
    arrays = d["arrays_%s" % sv]

    for ar in arrays:
        win_T = so_map.read_map(d["window_T_%s_%s" % (sv, ar)])
        win_pol = so_map.read_map(d["window_pol_%s_%s" % (sv, ar)])

        window_tuple = (win_T, win_pol)
        
        maps = d["maps_%s_%s" % (sv, ar)]
        nsplit[sv] = len(maps)

        cal = d["cal_%s_%s" % (sv, ar)]
        print("%s split of survey: %s, array %s"%(nsplit[sv], sv, ar))
        
        if deconvolve_pixwin:
            # ok so this is a bit overcomplicated because we need to take into account CAR and HEALPIX
            # for CAR the pixel window function deconvolution is done in Fourier space and take into account
            # the anisotropy if the pixwin
            # In HEALPIX it's a simple 1d function in multipole space
            # we also need to take account the case where we have projected Planck into a CAR pixellisation since
            # the native pixel window function of Planck need to be deconvolved
            if win_T.pixel == "CAR":
                wy, wx = enmap.calc_window(win_T.data.shape)
                inv_pixwin_lxly = (wy[:,None] * wx[None,:]) ** (-1)
                pixwin_l[sv] = np.ones(2 * lmax)
                if sv == "Planck":
                    print("Deconvolve Planck pixel window function")
                    # we include this special case for Planck projected in CAR taking into account the Planck native pixellisation
                    # we should check if the projection doesn't include an extra pixel window
                    inv_pixwin_lxly = None
                    pixwin_l[sv] = hp.pixwin(2048)

            elif win_T.pixel == "HEALPIX":
                pixwin_l[sv] = hp.pixwin(win_T.nside)
                
        else:
            inv_pixwin_lxly = None


        t = time.time()
        for k, map in enumerate(maps):
        
            if win_T.pixel == "CAR":
            
                split = so_map.read_map(map, geometry=win_T.data.geometry)
                
                if d["src_free_maps_%s" % sv] == True:
                    point_source_map_name = map.replace("srcfree.fits", "model.fits")
                    if point_source_map_name == map:
                        raise ValueError("No model map is provided! Check map names!")
                    point_source_map = so_map.read_map(point_source_map_name)
                    point_source_mask = so_map.read_map(d["ps_mask"])
                    split = data_analysis_utils.get_coadded_map(split, point_source_map, point_source_mask)

                split_copy = split.copy()
                split_copy = split_copy.downgrade(4)
                split_copy.plot(file_name="%s/no_filter_split_%s_%s_%d" % (plot_dir, sv, ar, k), color_range=color_range)

                
                if d["use_kspace_filter"]:
                    print("apply kspace filter on %s" %map)
                    binary = so_map.read_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
                    split = data_analysis_utils.get_filtered_map(
                        split, binary, vk_mask=d["vk_mask"], hk_mask=d["hk_mask"], normalize=False, inv_pixwin_lxly=inv_pixwin_lxly)
                else:
                    print("WARNING: no kspace filter is applied")
                    if deconvolve_pixwin:
                        print("WARNING: pixwin deconvolution in CAR WITHOUT kspace filter is not implemented")
                        sys.exit()
                        
            elif win_T.pixel == "HEALPIX":
                split = so_map.read_map(map)
                
            split.data *= cal
            if d["remove_mean"] == True:
                split = data_analysis_utils.remove_mean(split, window_tuple, ncomp)
            
            if d["use_kspace_filter"]:
                split.data /= (split.data.shape[1]*split.data.shape[2])
                
            split = split.downgrade(4)
            split.plot(file_name="%s/split_%s_%s_%d" % (plot_dir, sv, ar, k), color_range=color_range)

            print(time.time()- t)

