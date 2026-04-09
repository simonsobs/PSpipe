"""
This script projects SPT and ACT maps and masks into a CAR template defined around SPT survey
"""

from pixell import enplot, enmap, reproject, utils
import numpy as np
from pspy import so_map
from os.path import join as opj
import os

save_dir = ""
spt_maps_dir = opj(save_dir, "maps/spt")
spt_mask_dir = opj(save_dir, "masks")

os.makedirs(spt_maps_dir, exist_ok=True)
os.makedirs(spt_mask_dir, exist_ok=True)

### Define project geometry (arbitrary box around SPT patch + few degrees)

print("create SPT CAR geometry")
box = [[-77, -62],[-35,60]] * utils.adeg
shape,wcs = enmap.geometry2(pos=box,res=0.5 * utils.arcmin,proj='car', variant='fejer1')
print(f"Project on geometry: {shape=}, {wcs=}")
template = enmap.ones(shape, wcs)
enmap.write_map(opj(save_dir, "SPT_CAR_template.fits"), template)


### Project SPT full maps

spt_maps_read_dir = "/global/cfs/cdirs/sobs/users/tlouis/spt_data/real_data_maps"
spt_full_maps_fn_list = [opj(spt_maps_read_dir, f"full/full_{freq}ghz.fits") for freq in ["095"]]

for spt_maps_fn in spt_full_maps_fn_list: 
    # Project maps
    maps_so = so_map.read_map(spt_maps_fn, fields_healpix=(0, 1, 2))
    maps = maps_so.data

    maps_proj = reproject.healpix2map(maps, shape, wcs, method='harm')    # For maps we use harm
    
    enmap.write_map(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5] + "_CAR.fits"), maps_proj)

    plot = enplot.get_plots(
        maps_proj, ticks=10, mask=0, downgrade=2, colorbar=True, range=(100, 30, 30)
    )
    enplot.write(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5] + "_CAR"), plot)
    
    # We also need ivar, we use a fudge factor to get close to the real ivar: here I use 1e-2 which corresponds to 5uk.arcmin depth
    fudge_ivar = 1e-2
    maps_so = so_map.read_map(spt_maps_fn, fields_healpix=(3))
    maps = maps_so.data * fudge_ivar

    maps_proj = reproject.healpix2map(maps, shape, wcs, method='harm')    # For maps we use harm
    
    enmap.write_map(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5].replace("full", "ivar") + "_CAR.fits"), maps_proj)

    plot = enplot.get_plots(
        maps_proj, ticks=10, mask=0, downgrade=2, colorbar=True, range=(1)
    )
    enplot.write(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5].replace("full", "ivar") + "_CAR"), plot)    


spt_masks_read_dir = "/global/cfs/cdirs/sobs/users/tlouis/spt_data/ancillary_products/generally_applicable"
spt_masks_fn_list = [
    opj(spt_masks_read_dir, "pixel_mask_apodized_borders_objects.fits"),
    opj(spt_masks_read_dir, "pixel_mask_apodized_borders_only.fits"),
    opj(spt_masks_read_dir, "pixel_mask_binary_borders_objects.fits"),
    opj(spt_masks_read_dir, "pixel_mask_binary_borders_only.fits"),
]

for spt_mask_fn in spt_masks_fn_list: 
    # Project masks
    maps_so = so_map.read_map(spt_mask_fn)
    maps = maps_so.data

    maps_proj = reproject.healpix2map(maps, shape, wcs, method='spline')    # For masks we use spline
    
    enmap.write_map(opj(spt_mask_dir, os.path.basename(spt_mask_fn)[:-5] + "_CAR.fits"), maps_proj)

    plot = enplot.get_plots(
        maps_proj, ticks=10, mask=0, downgrade=2, colorbar=True, range=1
    )
    enplot.write(opj(spt_mask_dir, os.path.basename(spt_mask_fn)[:-5] + "_CAR"), plot)
    
    # We also need reverted masks for dory
    enmap.write_map(opj(spt_mask_dir, os.path.basename(spt_mask_fn)[:-5] + "_CAR_rev.fits"), 1 - maps_proj)

    plot = enplot.get_plots(
        1 - maps_proj, ticks=10, mask=0, downgrade=2, colorbar=True, range=1
    )
    enplot.write(opj(spt_mask_dir, os.path.basename(spt_mask_fn)[:-5] + "_CAR_rev"), plot)



