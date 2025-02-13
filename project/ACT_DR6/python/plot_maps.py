"""
This script compute all alms write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the CAR pixel window function.
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_map_preprocessing, so_mpi
from pspipe_utils import pspipe_list, kspace, misc, log
from pixell import enmap
import numpy as np
import healpy as hp
import sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
deconvolve_pixwin = d["deconvolve_pixwin"]
apply_kspace_filter = d["apply_kspace_filter"]

plot_dir = "plots/maps/"

pspy_utils.create_directory(plot_dir)

n_ar, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)

color_range = [500, 150, 150]

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
    
    win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
    win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
    
    window_tuple = (win_T, win_pol)

    if win_T.pixel == "CAR":
        win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])

        if apply_kspace_filter:
            ks_f = d[f"k_filter_{sv}"]
            filter = kspace.get_kspace_filter(win_T, ks_f)
                    
        if (deconvolve_pixwin == True):
            # deconvolve the CAR pixel function in fourier space
            wy, wx = enmap.calc_window(win_T.data.shape)
            inv_pixwin_lxly = (wy[:,None] * wx[None,:]) ** (-1)
        else:
            inv_pixwin_lxly = None
            
            
    maps = d[f"maps_{sv}_{ar}"]
    cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]

    for k, map in enumerate(maps):
        
        if win_T.pixel == "CAR":
            split = so_map.read_map(map, geometry=win_T.data.geometry)
                
            if d[f"src_free_maps_{sv}"] == True:
            
                
                ps_map_name = map.replace("srcfree.fits", "model.fits")
                if ps_map_name == map:
                    raise ValueError("No model map is provided! Check map names!")
                ps_map = so_map.read_map(ps_map_name)
                ps_mask = so_map.read_map(d[f"ps_mask_{sv}_{ar}"])
                ps_map.data *= ps_mask.data
                split.data += ps_map.data


            down_split = split.copy()
            down_split = down_split.downgrade(4)
            down_split.plot(file_name=f"{plot_dir}/no_filter_split_{sv}_{ar}_{k}", color_range=color_range)


            if apply_kspace_filter:
                log.info(f"apply kspace filter on {map}")
                split = kspace.filter_map(split,
                                          filter,
                                          win_kspace,
                                          inv_pixwin=inv_pixwin_lxly,
                                          weighted_filter=ks_f["weighted"])
                        
            else:
                log.info("WARNING: no kspace filter is applied")
                if deconvolve_pixwin:
                    split = so_map.fourier_convolution(split,
                                                       inv_pixwin_lxly,
                                                       window=win_kspace)
                         
        elif win_T.pixel == "HEALPIX":
            split = so_map.read_map(map)
                
        split = split.calibrate(cal=cal, pol_eff=pol_eff)
            
        if d["remove_mean"] == True:
            split = split.subtract_mean(window_tuple)

        down_split = split.copy()
        down_split = down_split.downgrade(4)
        down_split.plot(file_name=f"{plot_dir}/split_{sv}_{ar}_{k}", color_range=color_range)

        split.data[:] *= win_T.data[:]
        down_split = split.copy()
        down_split = down_split.downgrade(4)
        down_split.plot(file_name=f"{plot_dir}/windowed_split_{sv}_{ar}_{k}", color_range=color_range)
