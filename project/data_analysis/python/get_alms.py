"""
This script compute all alms write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the CAR pixel window function.
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_map_preprocessing, so_mpi
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
deconvolve_pixwin = d["deconvolve_pixwin"]

window_dir = "windows"
alms_dir = "alms"

pspy_utils.create_directory(alms_dir)

sv_list, ar_list = [], []
n_ar = 0
for sv in surveys:
    for ar in  d["arrays_%s" % sv]:
        sv_list += [sv]
        ar_list += [ar]
        n_ar += 1

print("number of arrays for the mpi loop : %s" % n_ar)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)
print(subtasks)

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
    
    win_T = so_map.read_map(d["window_T_%s_%s" % (sv, ar)])
    win_pol = so_map.read_map(d["window_pol_%s_%s" % (sv, ar)])
    window_tuple = (win_T, win_pol)

    ks_f = d["k_filter_%s" % sv]
    if win_T.pixel == "CAR" and ks_f["apply"]:
        shape, wcs = win_T.data.shape, win_T.data.wcs
        
        if ks_f["type"] == "binary_cross":
            filter = so_map_preprocessing.build_std_filter(shape, wcs, vk_mask=ks_f["vk_mask"], hk_mask=ks_f["hk_mask"], dtype=np.float32)
        elif ks_f["type"] == "gauss":
            filter = so_map_preprocessing.build_sigurd_filter(shape, wcs, ks_f["lbounds"], dtype=np.float32)
        else:
            print("you need to specify a valid filter type")
            sys.exit()
            
    if deconvolve_pixwin:
        # deconvolve the CAR pixel function in fourier space
        if win_T.pixel == "CAR":
            wy, wx = enmap.calc_window(win_T.data.shape)
            inv_pixwin_lxly = (wy[:,None] * wx[None,:]) ** (-1)
        else:
            inv_pixwin_lxly = None
            
    maps = d["maps_%s_%s" % (sv, ar)]
    cal = d["cal_%s_%s" % (sv, ar)]
            
    t = time.time()
    for k, map in enumerate(maps):
        
        if win_T.pixel == "CAR":
            split = so_map.read_map(map, geometry=win_T.data.geometry)
                
            if d["src_free_maps_%s" % sv] == True:
                point_source_map_name = map.replace("srcfree.fits", "model.fits")
                if point_source_map_name == map:
                    raise ValueError("No model map is provided! Check map names!")
                point_source_map = so_map.read_map(point_source_map_name)
                point_source_mask = so_map.read_map(d["ps_mask_%s_%s" % (sv, ar)])
                split = data_analysis_utils.get_coadded_map(split, point_source_map, point_source_mask)

            if ks_f["apply"]:
                print("apply kspace filter on %s" %map)
                binary = so_map.read_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
                norm, split = data_analysis_utils.get_filtered_map(split,
                                                                   binary,
                                                                   filter,
                                                                   inv_pixwin_lxly=inv_pixwin_lxly,
                                                                   weighted_filter=ks_f["weighted"])
                        
            else:
                print("WARNING: no kspace filter is applied")
                if deconvolve_pixwin:
                    binary = so_map.read_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
                    norm, split = data_analysis_utils.deconvolve_pixwin_CAR(split,
                                                                            binary,
                                                                            inv_pixwin_lxly)
                        
        elif win_T.pixel == "HEALPIX":
            split = so_map.read_map(map)
                
        split.data *= cal
            
        if d["remove_mean"] == True:
            split = data_analysis_utils.remove_mean(split, window_tuple, ncomp)
                
        master_alms = sph_tools.get_alms(split, window_tuple, niter, lmax)
            
        if ks_f["apply"] or deconvolve_pixwin:
            # there is an extra normalisation for the FFT/IFFT bit
            # note that we apply it here rather than at the FFT level because correcting the alm is faster than correcting the maps
            master_alms /= (split.data.shape[1] * split.data.shape[2]) ** norm
        
        np.save("%s/alms_%s_%s_%d.npy" % (alms_dir, sv, ar, k), master_alms)
        
    print(time.time()- t)





