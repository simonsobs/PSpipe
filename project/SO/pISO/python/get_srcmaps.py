"""Subtract sources from given maps and copy both initial maps and srcfree maps.
TODO : get actual source subtracted maps and don't use this script anymore :)
TODO : remove hardcoded paths (rn it should work for everyone on tiger though)
"""

import sys
import time
import os
import numpy as np
from pixell import enmap, enplot
from pspipe_utils import kspace, log, misc, pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, sph_tools

print(sys.argv)
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

template = enmap.read_map("/scratch/gpfs/SIMONSOBS/users/md9241/LAT_ISO/maps/zeros_deep56.fits")
shape, wcs = template.geometry

maps_path = "/scratch/gpfs/SIMONSOBS/lat-iso/phase2/deep56/20260201_inpaint_holes"
log.info("load f090 maps")
map_f090 = np.mean(
    [
        enmap.read_map(
            f"{maps_path}/deep56_full_{tube}_4way_0{s}_f090_sky_map.fits"
        )
        for s in range(4)
        for tube in ["i1", "i3", "i4", "i6"]
    ],
    axis=0,
)
log.info("load f090 srcfree maps")
map_f090_srcfree = np.mean(
    [
        enmap.read_map(
            f"{maps_path}/deep56_full_{tube}_4way_0{s}_f090_sky_map_srcfree.fits"
        )
        for s in range(4)
        for tube in ["i1", "i3", "i4", "i6"]
    ],
    axis=0,
)
log.info("load f150 maps")
map_f150 = np.mean(
    [
        enmap.read_map(
            f"{maps_path}/deep56_full_{tube}_4way_0{s}_f150_sky_map.fits"
        )
        for s in range(4)
        for tube in ["i1", "i3", "i4", "i6"]
    ],
    axis=0,
)
log.info("load f150 srcfree maps")
map_f150_srcfree = np.mean(
    [
        enmap.read_map(
            f"{maps_path}/deep56_full_{tube}_4way_0{s}_f150_sky_map_srcfree.fits"
        )
        for s in range(4)
        for tube in ["i1", "i3", "i4", "i6"]
    ],
    axis=0,
)

srcmap_f090 = enmap.ndmap(map_f090 - map_f090_srcfree, wcs)
srcmap_f150 = enmap.ndmap(map_f150 - map_f150_srcfree, wcs)

# save_path = maps_path
save_path = "/scratch/gpfs/SIMONSOBS/users/md9241/LAT_ISO/maps/srcmaps_20260201/"

enmap.write_map(save_path + f"deep56_full_coadd_4way_f090_sky_map_srcs.fits", srcmap_f090)
plot = enplot.get_plots(
    srcmap_f090, range=(1000, 300, 300), ticks=20, mask=0, downgrade=4, colorbar=True
)
enplot.write(save_path + f"deep56_full_coadd_4way_f090_sky_map_srcs", plot)

enmap.write_map(save_path + f"deep56_full_coadd_4way_f150_sky_map_srcs.fits", srcmap_f150)
plot = enplot.get_plots(
    srcmap_f150, range=(1000, 300, 300), ticks=20, mask=0, downgrade=4, colorbar=True
)
enplot.write(save_path + f"deep56_full_coadd_4way_f150_sky_map_srcs", plot)