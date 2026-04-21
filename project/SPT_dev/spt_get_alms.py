"""
This script compute all alms write them to disk.
It uses the window function provided in the dictionnary file and optionnaly applies an alm mask.
"""

import sys
import time

import numpy as np
import healpy as hp
from pixell import enmap
from pspipe_utils import log, pspipe_list
from pspy import pspy_utils, so_dict, so_mpi, sph_tools, so_map, so_window
import sph_tools_mod




d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
apply_alm_mask = d["apply_alm_mask"]
pure = d["pure"]


alms_dir = "alms"
pspy_utils.create_directory(alms_dir)
if pure == True:
    plot_dir = "plots"
    pspy_utils.create_directory(plot_dir)

n_ar, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)


for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
    alm_conv = d[f"alm_conv_{sv}"]

    log.info(f"[{task}] Computing alm for '{sv}' survey and '{ar}' array")

    win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
    win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
    
    if pure == True:
        spinned_windows = so_window.get_spinned_windows(win_pol, lmax, niter=niter)
        w1_plus, w1_minus, w2_plus, w2_minus = spinned_windows
        w1_plus.plot(file_name=f"{plot_dir}/win_spin1_a_{sv}_{ar}")
        w1_minus.plot(file_name=f"{plot_dir}/win_spin1_b_{sv}_{ar}")
        w2_plus.plot(file_name=f"{plot_dir}/win_spin2_a_{sv}_{ar}")
        w2_minus.plot(file_name=f"{plot_dir}/win_spin2_b_{sv}_{ar}")

    
    window_tuple = (win_T, win_pol)

    maps = d[f"maps_{sv}_{ar}"]
    cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]

    t0 = time.time()
    for k, map in enumerate(maps):

        split = so_map.read_map(map)
        split = split.calibrate(cal=cal, pol_eff=pol_eff)

        if d["remove_mean"] == True:
            split = split.subtract_mean(window_tuple)

        if pure == False:
            master_alms = sph_tools.get_alms(split, window_tuple, niter, lmax, alm_conv=alm_conv)
        else:
            master_alms = sph_tools_mod.get_pure_alms(split, window_tuple, spinned_windows, niter, lmax, alm_conv=alm_conv)

        
        if apply_alm_mask == True:
            alm_mask = hp.read_alm(d[f"alm_mask_{sv}_{ar}"], hdu=1)
            alm_mask = hp.sphtfunc.resize_alm(alm_mask, d["lmax_mask"], d["lmax_mask"], lmax, lmax)
            master_alms *= alm_mask
                
        np.save(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy", master_alms)

    log.info(f"[{task}] execution time {time.time() - t0} seconds")
