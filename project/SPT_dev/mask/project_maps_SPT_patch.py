"""
This script projects SPT  maps and masks into a CAR template defined around SPT survey
"""

from pixell import enplot, enmap, reproject, utils
import numpy as np
from pspy import so_map, so_dict
from os.path import join as opj
import os, sys


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

release_dir = d["release_dir"]
freqs = d["freqs_spt"]

save_dir = ""
spt_maps_dir = opj(save_dir, "projected_maps/spt")
spt_mask_dir = opj(save_dir, "projected_masks")

os.makedirs(spt_maps_dir, exist_ok=True)
os.makedirs(spt_mask_dir, exist_ok=True)

### Define project geometry (arbitrary box around SPT patch + few degrees)

print("create SPT CAR geometry")
box = d["box"] * utils.adeg
shape,wcs = enmap.geometry2(pos=box, res=0.5 * utils.arcmin,proj='car', variant='fejer1')
print(f"Project on geometry: {shape=}, {wcs=}")


### Project SPT full maps

spt_maps_read_dir = f"{release_dir}/real_data_maps"
spt_full_maps_fn_list = [opj(spt_maps_read_dir, f"full/full_{freq}ghz.fits") for freq in freqs]


i_var_norm = [0.2638, 0.3865, 0.0308] # From  Wei Quan email, unit are 1/(\mu K^{2}/Hz)

for count, spt_maps_fn in enumerate(spt_full_maps_fn_list):
    # Project maps
    maps_so = so_map.read_map(spt_maps_fn, fields_healpix=(0, 1, 2))
    maps = maps_so.data

    maps_proj = reproject.healpix2map(maps, shape, wcs, method="harm")    # For maps we use harm
    
    enmap.write_map(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5] + "_CAR.fits"), maps_proj)

    plot = enplot.get_plots(
        maps_proj, ticks=10, mask=0, colorbar=True, range=(100, 30, 30)
    )
    enplot.write(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5] + "_CAR"), plot)
    
    maps_so = so_map.read_map(spt_maps_fn, fields_healpix=(3))
    maps = maps_so.data * i_var_norm[count]

    maps_proj = reproject.healpix2map(maps, shape, wcs, method="spline")
    # check if there is negative pixel
    n_neg = (maps_proj < 0).sum()
    if n_neg > 0:
        print(f"Warning : {n_neg} negative pixels after reprojection (min: {maps_proj.min():.2e})")
        maps_proj = np.maximum(maps_proj, 0)
    
    enmap.write_map(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5].replace("full", "ivar") + "_CAR.fits"), maps_proj)

    plot = enplot.get_plots(
        maps_proj, ticks=10, mask=0, colorbar=True, range=(10**-2)
    )
    enplot.write(opj(spt_maps_dir, os.path.basename(spt_maps_fn)[:-5].replace("full", "ivar") + "_CAR"), plot)    


spt_masks_read_dir = f"{release_dir}/ancillary_products/generally_applicable"
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
    
    if "binary" in spt_mask_fn:
        # make sure the mask remains binary despite projection effect
        maps_proj[maps_proj > 0.5] = 1
        maps_proj[maps_proj < 0.5] = 0

    
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



