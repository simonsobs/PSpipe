"""
This script compute all alms write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the CAR pixel window function.
"""

import sys
import time

import numpy as np
from pixell import enmap
from pspipe_utils import kspace, log, misc, pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, sph_tools

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
deconvolve_pixwin = d["deconvolve_pixwin"]
niter = d["niter"]
apply_kspace_filter = d["apply_kspace_filter"]


alms_dir = "alms"
pspy_utils.create_directory(alms_dir)

n_ar, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]

    log.info(f"[{task}] Computing alm for '{sv}' survey and '{ar}' array")
    maps = d[f"maps_{sv}_{ar}"]

    t0 = time.time()
    for k, map in enumerate(maps):
    
        cal, pol_eff = d[f"cal_{sv}_{ar}_per_split"][k], d[f"pol_eff_{sv}_{ar}_per_split"][k]

        win_T = so_map.read_map(d[f"window_T_{sv}_{ar}_per_split"][k])
        win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}_per_split"][k])

        window_tuple = (win_T, win_pol)

        if win_T.pixel == "CAR":

            inv_pixwin_lxly = None
            if deconvolve_pixwin:
                if d[f"pixwin_{sv}"]["pix"] == "CAR":
                    # compute the CAR pixel function in fourier space
                    wy, wx = enmap.calc_window(win_T.data.shape, order=d[f"pixwin_{sv}"]["order"])
                    inv_pixwin_lxly = (wy[:,None] * wx[None,:]) ** (-1)

            split = so_map.read_map(map, geometry=win_T.data.geometry)

            if d[f"src_free_maps_{sv}"] == True:
                ps_map_name = map.replace("srcfree.fits", "model.fits")
                if ps_map_name == map:
                    raise ValueError("No model map is provided! Check map names!")
                ps_map = so_map.read_map(ps_map_name)
                ps_mask = so_map.read_map(d[f"ps_mask_{sv}_{ar}"])
                ps_map.data *= ps_mask.data
                split.data += ps_map.data

            if apply_kspace_filter:
            
                win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}_per_split"][k])

                ks_f = d[f"k_filter_{sv}"]
                filter = kspace.get_kspace_filter(win_T, ks_f)

                log.info(f"[{task}] apply kspace filter on {map}")
                split = kspace.filter_map(split,
                                          filter,
                                          win_kspace,
                                          inv_pixwin=inv_pixwin_lxly,
                                          weighted_filter=ks_f["weighted"],
                                          use_ducc_rfft=True)
                        
            else:
                log.info(f"[{task}] WARNING: no kspace filter is applied")
                if (deconvolve_pixwin) & (inv_pixwin_lxly is not None):
                    split = so_map.fourier_convolution(split,
                                                       inv_pixwin_lxly,
                                                       window=win_kspace,
                                                       use_ducc_rfft=True)
                         
        elif win_T.pixel == "HEALPIX":
            split = so_map.read_map(map, fields_healpix=(0, 1, 2))

        split = split.calibrate(cal=cal, pol_eff=pol_eff)

        if d["remove_mean"] == True:
            split = split.subtract_mean(window_tuple)
        if d["remove_mono_dipole"] == True:
            split.subtract_mono_dipole(window_tuple)

        master_alms = sph_tools.get_alms(split, window_tuple, niter, lmax)
        np.save(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy", master_alms)

    log.info(f"[{task}] execution time {time.time() - t0} seconds")
