"""
Compute masks from ivar (smooth+threshold on non-zeros percentile) of SO LAT maps.
Optionnally plots all ivar and maps
"""
from pspy import so_map, so_dict, so_window, so_mpi
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

additionnal_mask = so_map.read_map('/global/cfs/cdirs/sobs/users/merrydup/LAT_ISO/masks_1024/act_xtra_mask_dr6_pa5_f090.fits')
additionnal_mask = None
if additionnal_mask is not None:
    log.info('Additionnal mask used')

def TQU2T(maps:so_map.so_map) -> so_map.so_map:
    maps_T = so_map.car_template_from_shape_wcs(1, maps.data.shape, maps.data.wcs)
    maps_T.data = maps.data[0]
    return maps_T

ivar_percentile = 20
ivar_percentile = 5

plot_maps_and_ivar = True
if plot_maps_and_ivar:
    save_path_maps_ivar = save_path + '/maps_ivar/'
    os.makedirs(save_path_maps_ivar, exist_ok=True)

n_ar = len(d['arrays_SO'])

log.info(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)

for task in subtasks:
    task = int(task)
    ar = d['arrays_SO'][task]
    maps_filenames = d[f'maps_SO_{ar}']
    log.info(f"[{task}] Computing mask for '{ar}' array")

    masks_dict[ar] = []
    for i, maps_filename in enumerate(maps_filenames):
        save_name = save_path + f'{ar}_split{i}_mask'
        
        if 'sky_div' in d.keys():
            ivar_filename = maps_filename.replace('pass2_sky_map', 'sky_div')
            ivar = so_map.read_map(ivar_filename)
            ivar = TQU2T(TQU2T(ivar))
        else:
            ivar_filename = maps_filename.replace('_map', '_ivar')
            ivar = so_map.read_map(ivar_filename)
        
        
        if plot_maps_and_ivar:
            save_name_maps_ivar = save_path_maps_ivar + f'{ar}_split{i}_mask'
            log.info(f'plot {ar} maps and ivar')
            maps = so_map.read_map(maps_filename).calibrate(d[f"cal_SO_{ar}"], d[f"pol_eff_SO_{ar}"])
            maps.downgrade(4).plot(file_name=save_name_maps_ivar[:-4] + 'map', color_range=(500, 100, 100))
            ivar.downgrade(4).plot(file_name=save_name_maps_ivar)
        
        mask = ivar.copy()
        mask.data *= 0.
        smoothed_ivar = enmap.smooth_gauss(ivar.data, np.radians(0.2))
        mask.data[np.where(smoothed_ivar >= np.percentile(ivar.data[np.nonzero(ivar.data)], ivar_percentile))] = 1.
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
    
    save_name = save_path + f'{ar}_inter_mask'
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

save_name = save_path + f'union_mask'
mask_union_arrays.downgrade(4).plot(file_name=save_name, color_range=color_range)
mask_union_arrays.write_map(file_name=save_name + '.fits')
log.info(f'union survey solid angle : {so_window.get_survey_solid_angle(mask_union_arrays):.5f}')

