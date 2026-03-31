"""
This script inpaints spt maps
"""

import sys
import spt_plot_helper as spt_plot
from pspy import so_dict
import healpy as hp
import numpy as np

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

survey = "spt"

arrays = d[f"arrays_{survey}"]
n_splits_spt = d["n_splits_spt"]
release_dir = d["release_dir"]

binary_border_object_mask_path = release_dir + "ancillary_products/generally_applicable/pixel_mask_binary_borders_objects.fits"
binary_border_object_mask = hp.read_map(binary_border_object_mask_path)

masked_pixels = np.where(binary_border_object_mask==0)

for ar in arrays:

    if n_splits_spt == 2:
        maps = [release_dir + "real_data_maps/half/half_bundle%d_%sghz.fits" % (i, ar) for i in range(n_splits_spt)]
    if n_splits_spt == 30:
        maps = [release_dir + "real_data_maps/one_thirtieth/one_thirtieth_bundle%02d_%sghz.fits" % (i, ar) for i in range(n_splits_spt)]

    for map_name in maps:
        inpainted_map = []
        for stokes, field in zip(["t", "q", "u"], [0, 1, 2]):
        
            m_before_inpainting = hp.read_map(map_name, field=field)
            
            inpainted_pixel_values_path = release_dir + f"ancillary_products/specific_to_c25/inpainted_pixel_values_{stokes}_{ar}ghz.npy"
            inpainted_pixel_values = np.load(inpainted_pixel_values_path)
            
            m_after_inpainting = np.array(m_before_inpainting)
            m_after_inpainting[masked_pixels] = inpainted_pixel_values
            
            inpainted_map += [m_after_inpainting]
           
        map_name = map_name.replace(".fits","")
        map_name_inpainted = map_name + "_inpainted.fits"
        hp.write_map(map_name_inpainted, inpainted_map, overwrite=True, partial=True)


