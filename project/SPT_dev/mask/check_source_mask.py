"""
This script projects SPT  maps and masks into a CAR template defined around SPT survey
"""

from pixell import reproject, utils, enmap
import numpy as np
from pspy import so_map, so_dict
from os.path import join as opj
import os, sys


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

n_tops = d["n_tops"]
box = d["box"] *  utils.adeg

shape, wcs = enmap.geometry2(pos=box, res=0.5 * utils.arcmin,proj='car', variant='fejer1')

mask_dir = "my_masks"

SPT_map = so_map.read_map("projected_maps/spt/full_095ghz_CAR.fits")
SPT_map.plot(file_name=f"{mask_dir}/spt", color_range=[250,30,30])


for n_top in n_tops:


    spt_mask = so_map.read_map(f"{mask_dir}/spt_source_mask_top{n_top}_apod.fits")
    spt_mask_proj = reproject.healpix2map(spt_mask.data, shape, wcs, method="spline")    # For maps we use harm


    SPT_map_copy = SPT_map.copy()

    SPT_map_copy.data *= spt_mask_proj
    SPT_map_copy.plot(file_name=f"{mask_dir}/mask_spt_{n_top}", color_range=[250,30,30])

