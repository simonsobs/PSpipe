"""
Compute masks from ivar (smooth+threshold on non-zeros percentile).
Optionnally plots ll ivar and maps
"""
from pspy import so_map, so_dict, so_window
import numpy as np
from pixell import enmap
import sys
from copy import deepcopy
import os
from pspipe_utils import log

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

save_path = d['mask_dir_xtra']
os.makedirs(save_path, exist_ok=True)
color_range = None

masks_dict = {}
mask_intersection_splits = {}

additionnal_mask = so_map.read_map('/pscratch/sd/m/merrydup/PSpipe_SO/masks/20251019/deep56_glitch_mask.fits')
additionnal_mask = None
if additionnal_mask is not None:
    log.info('Additionnal mask used')

plot_maps_and_ivar = True
if plot_maps_and_ivar:
    save_path_maps_ivar = save_path + '/maps_ivar/'
    os.makedirs(save_path_maps_ivar, exist_ok=True)

for ar in d['arrays_SO']:  
    maps_filenames = d[f'maps_SO_{ar}']
    
    masks_dict[ar] = []
    for i, maps_filename in enumerate(maps_filenames):
        save_name = save_path + f'deep56_{ar}_split{i}_mask'
        
        ivar_filename = maps_filename.replace('_map', '_ivar')
        ivar = so_map.read_map(ivar_filename)
        
        if plot_maps_and_ivar:
            save_name_maps_ivar = save_path_maps_ivar + f'deep56_{ar}_split{i}_mask'
            log.info(f'plot {ar} maps and ivar')
            maps = so_map.read_map(maps_filename)
            maps.downgrade(2).plot(file_name=save_name_maps_ivar[:-4] + 'map', color_range=(500, 100, 100))
            ivar.downgrade(2).plot(file_name=save_name_maps_ivar)
        
        mask = ivar.copy()
        mask.data *= 0.
        smoothed_ivar = enmap.smooth_gauss(ivar.data, np.radians(0.2))
        mask.data[np.where(smoothed_ivar >= np.percentile(ivar.data[np.nonzero(ivar.data)], 20))] = 1.
        if additionnal_mask is not None:
            mask.data *= additionnal_mask.data
        mask.downgrade(4).plot(file_name=save_name, color_range=color_range)
        mask.write_map(file_name=save_name + '.fits')
        
        log.info(f'{ar} split {i} survey solid angle : {so_window.get_survey_solid_angle(mask):.5f}')
        
        masks_dict[ar].append(mask)
    
    # Initialize a mask for intersection of splits of a same array
    mask_intersection_splits[ar] = deepcopy(masks_dict[ar][0])
    mask_intersection_splits[ar].data = 1.
    for mask in masks_dict[ar]:
        mask_intersection_splits[ar].data *= mask.data
    
    save_name = save_path + f'deep56_{ar}_inter_mask'
    mask_intersection_splits[ar].downgrade(4).plot(file_name=save_name, color_range=color_range)
    mask_intersection_splits[ar].write_map(file_name=save_name + '.fits')
    log.info(f'{ar} intersection survey solid angle : {so_window.get_survey_solid_angle(mask_intersection_splits[ar]):.5f}')
    
log.info(f'Make union mask')

# Initialize a mask for UNION of all arrays for ACT/Planck data
mask_union_arrays = deepcopy(masks_dict[ar][0])
mask_union_arrays.data = 1.
for mask in mask_intersection_splits.values():
    mask_union_arrays.data *= (1. - mask.data)
mask_union_arrays.data = (1. - mask_union_arrays.data)

save_name = save_path + f'deep56_union_mask'
mask_union_arrays.downgrade(4).plot(file_name=save_name, color_range=color_range)
mask_union_arrays.write_map(file_name=save_name + '.fits')
log.info(f'union survey solid angle : {so_window.get_survey_solid_angle(mask_union_arrays):.5f}')

