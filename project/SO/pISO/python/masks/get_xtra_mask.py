"""
Compute masks from ivar (smooth+threshold on non-zeros percentile).
Optionally plots all ivar and maps
"""

from pspy import so_dict, so_map, so_window, pspy_utils
from pspipe_utils import log

from pixell import enmap, enplot      

import numpy as np
import yaml

import os
from os.path import join as opj
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# log mask infos from mask yaml file
with open(d['xtra_mask_yaml'], "r") as f:
    mask_dict: dict = yaml.safe_load(f)
mask_infos = mask_dict['get_xtra_mask.py']

mask_dir = d['mask_dir']

save_plot_mask = mask_infos['save_plot_mask'] # should we make and save the plots of masks?
save_plot_maps_ivar = mask_infos['save_plot_maps_ivar'] # should we save the plotted maps and ivars?

if save_plot_mask or save_plot_maps_ivar:
    plot_dir_mask = opj(d['plots_dir'], 'mask')
    pspy_utils.create_directory(plot_dir_mask)

if save_plot_maps_ivar:
    plot_dir_map_ivar = opj(plot_dir_mask, 'maps_ivar')
    pspy_utils.create_directory(plot_dir_map_ivar)

# get reasonable ivar, top 90% of nonzero values seems to work decently
ivar_smooth_deg = mask_infos['ivar_smooth_deg']
ivar_quantile = mask_infos['ivar_quantile']

mask_intersect = True # intersection of every single mask
mask_union = False # union of every single mask

# there may be some extra xtra mask that we want to also include for
# every single mask
additional_mask_fn = mask_infos.get('additional_mask')
if additional_mask_fn is not None:
    log.info(f'Additional mask used: {additional_mask_fn}')
    additional_mask = enmap.read_map(additional_mask_fn).astype(bool)
else:
    additional_mask = True

for sv in mask_infos['surveys_to_xtra_mask']:
    for m in d[f'arrays_{sv}']:
        ivar_mask = True # intersection of masks just for this map (over splits)
        
        map_fns = d[f'maps_{sv}_{m}']
        for i, map_fn in enumerate(map_fns):

            map_dir_fn, map_base_fn = os.path.split(map_fn)
            if d[f"src_free_maps_{sv}"] == True:
                ivar_fn = map_fn.replace('_map_srcfree', '_ivar')
            else:
                ivar_fn = map_fn.replace('_map', '_ivar')   

            # mask is based on the smoothed ivar map
            # only the pixels where the original ivar were nonzero though
            ivar = enmap.read_map(ivar_fn)
            ivar = ivar.reshape(-1, *ivar.shape[-2:])[0] # the "first" ivar map

            ivar_smooth = enmap.smooth_gauss(ivar, np.deg2rad(ivar_smooth_deg))
            ivar_set_mask = ivar_smooth > np.quantile(ivar_smooth[ivar > 0], ivar_quantile)
            ivar_set_mask *= ivar > 0
            ivar_set_mask = np.logical_and(ivar_set_mask, additional_mask)
            
            # check that inside of ivar_mask, there are no zero ivar
            assert np.all(ivar[ivar_set_mask] > 0), \
                f'{sv}, {m}, set{i} has zero ivar inside ivar_mask'
        
            # possibly plot maps and ivars
            if save_plot_maps_ivar:
                log.info(f'plot {sv}, {m}, set{i} map and ivar')

                map = enmap.read_map(map_fn)
                p = enplot.plot(map, downgrade=8, ticks=1, colorbar=True, range=[500, 100, 100])
                map_plot_fn = os.path.splitext(os.path.basename(map_fn))[0]
                enplot.write(opj(plot_dir_map_ivar, map_plot_fn), p)

                p = enplot.plot(ivar, downgrade=8, ticks=1, colorbar=True)
                ivar_plot_fn = os.path.splitext(os.path.basename(ivar_fn))[0]
                enplot.write(opj(plot_dir_map_ivar, ivar_plot_fn), p)

            log.info(f'{sv}, {m}, set{i} survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_set_mask)) / (4 * np.pi) * 41253:.5f}')

            # save mask
            ivar_set_mask_fn = opj(mask_dir, f'xtra_mask_{sv}_{m}_set{i}.fits')
            enmap.write_map(ivar_set_mask_fn, ivar_set_mask.astype(np.float32))

            # save plot of mask
            if save_plot_mask:
                p = enplot.plot(ivar_set_mask, downgrade=8, ticks=1, colorbar=True)
                mask_plot_fn = os.path.splitext(os.path.basename(ivar_set_mask_fn))[0]
                enplot.write(opj(plot_dir_mask, mask_plot_fn), p)

            # also build xtra masks that are the union and intersection
            # of all the xtra masks
            mask_intersect = np.logical_and(mask_intersect, ivar_set_mask)
            mask_union = np.logical_or(mask_union, ivar_set_mask)
            ivar_mask = np.logical_and(ivar_mask, ivar_set_mask)

        log.info(f'{sv}, {m} intersection survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_mask)) / (4 * np.pi) * 41253:.5f}')

        # save mask
        ivar_mask_fn = opj(mask_dir, f'xtra_mask_{sv}_{m}.fits')
        enmap.write_map(ivar_mask_fn, ivar_mask.astype(np.float32))

        # save plot of mask
        if save_plot_mask:
            p = enplot.plot(ivar_mask, downgrade=8, ticks=1, colorbar=True)
            mask_plot_fn = os.path.splitext(os.path.basename(ivar_mask_fn))[0]
            enplot.write(opj(plot_dir_mask, mask_plot_fn), p)
            
log.info(f'All intersection survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(mask_intersect)) / (4 * np.pi) * 41253:.5f}')
log.info(f'All union survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(mask_union)) / (4 * np.pi) * 41253:.5f}')

# plot and save union and intersect masks
p = enplot.plot(mask_intersect, downgrade=8, ticks=1, colorbar=True)
enplot.write(opj(plot_dir_mask, f'xtra_mask_intersect'), p)
enmap.write_map(opj(mask_dir, f'xtra_mask_intersect.fits'), mask_intersect.astype(np.float32))

p = enplot.plot(mask_union, downgrade=8, ticks=1, colorbar=True)
enplot.write(opj(plot_dir_mask, f'xtra_mask_union'), p)
enmap.write_map(opj(mask_dir, f'xtra_mask_union.fits'), mask_union.astype(np.float32))
