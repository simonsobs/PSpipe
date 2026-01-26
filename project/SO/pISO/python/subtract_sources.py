"""Subtract sources from given maps and copy both initial maps and srcfree maps.
TODO : get actual source subtracted maps and don't use this script anymore :)
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

surveys_to_subtract = ["lat_iso"]  # Only work with LAT ISO
d["surveys"] = surveys_to_subtract

n_ar, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar - 1)

act_map_path = d["maps_dir_dr6"]
map_f090 = np.mean(
    [
        enmap.read_map(
            f"{act_map_path}/act_dr6.02_std_AA_night_{tube}_f090_4way_set{s}_map.fits"
        )
        for s in range(4)
        for tube in ["pa5", "pa6"]
    ],
    axis=0,
)
map_f090_srcfree = np.mean(
    [
        enmap.read_map(
            f"{act_map_path}/act_dr6.02_std_AA_night_{tube}_f090_4way_set{s}_map_srcfree.fits"
        )
        for s in range(4)
        for tube in ["pa5", "pa6"]
    ],
    axis=0,
)
map_f150 = np.mean(
    [
        enmap.read_map(
            f"{act_map_path}/act_dr6.02_std_AA_night_{tube}_f150_4way_set{s}_map.fits"
        )
        for s in range(4)
        for tube in ["pa5", "pa6"]
    ],
    axis=0,
)
map_f150_srcfree = np.mean(
    [
        enmap.read_map(
            f"{act_map_path}/act_dr6.02_std_AA_night_{tube}_f150_4way_set{s}_map_srcfree.fits"
        )
        for s in range(4)
        for tube in ["pa5", "pa6"]
    ],
    axis=0,
)
map_f220 = np.mean(
    [
        enmap.read_map(
            f"{act_map_path}/act_dr6.02_std_AA_night_pa4_f220_4way_set{s}_map.fits"
        )
        for s in range(4)
    ],
    axis=0,
)
map_f220_srcfree = np.mean(
    [
        enmap.read_map(
            f"{act_map_path}/act_dr6.02_std_AA_night_pa4_f220_4way_set{s}_map_srcfree.fits"
        )
        for s in range(4)
    ],
    axis=0,
)

srcmap_f090 = map_f090 - map_f090_srcfree
srcmap_f150 = map_f150 - map_f150_srcfree
srcmap_f220 = map_f220 - map_f220_srcfree

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
    for i, map_fn in enumerate(d[f"maps_{sv}_{ar}"]):
        if d["src_free_maps_lat_iso"]:
            map_fn = map_fn.replace("_srcfree", "")
        log.info(f"{sv}_{ar}_split{i}")
        input_map = enmap.read_map(map_fn)
        lat_dir = os.path.dirname(map_fn)

        if ("f090" in map_fn) or ("f100" in map_fn):
            srcmap = srcmap_f090
        elif ("f150" in map_fn) or ("f143" in map_fn):
            srcmap = srcmap_f150
        elif ("f220" in map_fn) or ("f280" in map_fn) or ("f217" in map_fn):
            srcmap = srcmap_f220

        srcfree_map = input_map - srcmap
        enmap.write_map(f"{map_fn[:-5]}_srcfree.fits", srcfree_map)
        enplot.write(
            f"{map_fn[:-5]}_srcfree.fits",
            enplot.plot(
                srcfree_map, downgrade=8, ticks=1, colorbar=True, range=(1000, 300, 300)
            ),
        )
