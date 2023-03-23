"""
This script compute all alms write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the CAR pixel window function.
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_map_preprocessing, so_mpi
from pspipe_utils import pspipe_list, kspace, misc
from pixell import enmap
import numpy as np
import healpy as hp
import sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
deconvolve_pixwin = d["deconvolve_pixwin"]
niter = d["niter"]
apply_kspace_filter = d["apply_kspace_filter"]


#window_dir = "windows"
window_dir = d["window_dir"]
alms_dir = "alms"

pspy_utils.create_directory(alms_dir)
        
n_ar, sv_list, ar_list = pspipe_list.get_arrays_list(d)

print(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)
print(subtasks)

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
    
    win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
    win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
    
    window_tuple = (win_T, win_pol)


    if win_T.pixel == "CAR":
        # this doesn't work when using the weighted window, window_w:
        #binary_file = misc.str_replace(d[f"window_T_{sv}_{ar}"], "window_", "binary_")
        #binary = so_map.read_map(binary_file)
        
        # I prefer how it was before:
        binary = so_map.read_map(f"{window_dir}/binary_{sv}_{ar}.fits")
        
        # same kfilter and pixwin for all seasons
        if apply_kspace_filter:
            #ks_f = d[f"k_filter_{sv}"]
            ks_f = d[f"k_filter"]
            filter = kspace.get_kspace_filter(win_T, ks_f)
                    
        inv_pixwin_lxly = None
        if deconvolve_pixwin:
            #if d[f"pixwin_{sv}"]["pix"] == "CAR":
            if d[f"pixwin"]["pix"] == "CAR":
                # compute the CAR pixel function in fourier space
                #wy, wx = enmap.calc_window(win_T.data.shape, order=d[f"pixwin_{sv}"]["order"])
                wy, wx = enmap.calc_window(win_T.data.shape, order=d[f"pixwin"]["order"])
                inv_pixwin_lxly = (wy[:,None] * wx[None,:]) ** (-1)
            
            
    maps = d[f"maps_{sv}_{ar}"]
    cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]

    t = time.time()
    for k, map in enumerate(maps):
        
        if win_T.pixel == "CAR":
            split = so_map.read_map(map, geometry=win_T.data.geometry)
                
            if d[f"src_free_maps_{sv}"] == True:
                ps_map_name = map.replace("map_srcfree.fits", "srcs.fits")
                if ps_map_name == map:
                    raise ValueError("No model map is provided! Check map names!")
                ps_map = so_map.read_map(ps_map_name)
                ps_mask = so_map.read_map(d[f"ps_mask"])
                ps_map.data *= ps_mask.data
                split.data += ps_map.data

            if apply_kspace_filter:
                print(f"apply kspace filter on {map}")
                split = kspace.filter_map(split,
                                          filter,
                                          binary,
                                          inv_pixwin=inv_pixwin_lxly,
                                          weighted_filter=ks_f["weighted"])
                        
            else:
                print("WARNING: no kspace filter is applied")
                if deconvolve_pixwin:
                    split = so_map.fourier_convolution(split,
                                                       inv_pixwin_lxly,
                                                       binary=binary)
                         
        elif win_T.pixel == "HEALPIX":
            split = so_map.read_map(map)
                
        split = split.calibrate(cal=cal, pol_eff=pol_eff)
            
        if d["remove_mean"] == True:
            split = split.subtract_mean(window_tuple)
                
        master_alms = sph_tools.get_alms(split, window_tuple, niter, lmax)
        
        np.save(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy", master_alms)
        
    print(time.time()- t)





