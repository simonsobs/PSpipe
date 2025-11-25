"""
Compute masks from ivar (smooth+threshold on non-zeros percentile).
Optionnally plots all ivar and maps
"""

from pspy import so_dict, so_map, so_window, so_mpi
from pspipe_utils import log

from pixell import enmap, enplot      

import numpy as np

import os
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

save_plot_mask = True # should we make and save the plots of masks?
save_plot_maps_ivar = True # should we save the plotted maps and ivars?

mask_dir = d['mask_dir']
if save_plot_maps_ivar:
    maps_ivar_dir = os.path.join(mask_dir, 'maps_ivar')
    os.makedirs(maps_ivar_dir, exist_ok=True)

# get reasonable ivar, top 90% of nonzero values seems to work decently
ivar_smooth_deg = d['ivar_smooth_deg']
ivar_quantile = d['ivar_quantile']

mask_intersect = True # intersection of every single mask
mask_union = False # union of every single mask

# there may be some extra xtra mask that we want to also include for
# every single mask
additional_mask_fn = d.get('additional_mask')
if additional_mask_fn is not None:
    log.info(f'Additional mask used: {additional_mask_fn}')
    additional_mask = enmap.read_map(additional_mask_fn).astype(bool)
else:
    additional_mask = True

n_ar = len(d['arrays_SO'])

log.info(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)

sv = 'SO'   # we want to run tthise script only on SO

for task in subtasks:
        m = d[f'arrays_{sv}'][task]
        ivar_mask = True # intersection of masks just for this map (over splits)
        
        map_fns = d[f'maps_{sv}_{m}']
        for i, map_fn in enumerate(map_fns):

            map_dir_fn, map_base_fn = os.path.split(map_fn)
            if d[f"src_free_maps_{sv}"] == True:
                ivar_fn = map_fn.replace('_map_srcfree', '_ivar')
            else:
                ivar_fn = map_fn.replace('_map', '_ivar')   

            # mask is based on the smoothed ivar map
            # only the pixels were the original ivar were nonzero though
            ivar = enmap.read_map(ivar_fn).astype(bool) 
            while len(ivar.shape) > 2:
                ivar = ivar[0].astype(np.float32)

            # ivar_set_mask = ivar_smooth > np.quantile(ivar_smooth[ivar > 0], ivar_quantile)
            # ivar_set_mask = np.logical_and(ivar_set_mask, additional_mask)

            ivar_smooth = enmap.smooth_gauss(ivar, np.deg2rad(ivar_smooth_deg))
            ivar_set_mask = enmap.zeros(ivar.shape, ivar.wcs, dtype=bool)
            ivar_set_mask[ivar_smooth > np.quantile(ivar_smooth[ivar > 0], ivar_quantile)] = 1
            # check that inside of ivar_mask, there are no zero ivar
            assert np.all(ivar[ivar_set_mask] > 0), \
                f'{sv}, {m}, set{i} has zero ivar inside ivar_mask'
        
            # possibly plot maps and ivars
            if save_plot_maps_ivar:
                log.info(f'plot {sv}, {m}, set{i} map and ivar')

                map = enmap.read_map(map_fn)
                p = enplot.plot(map, downgrade=8, ticks=1, colorbar=True, range=[500, 100, 100])
                map_plot_fn = os.path.basename(map_fn)
                enplot.write(os.path.join(maps_ivar_dir, map_plot_fn)[:-5], p)

                p = enplot.plot(ivar, downgrade=8, ticks=1, colorbar=True)
                ivar_plot_fn = os.path.basename(ivar_fn)
                enplot.write(os.path.join(maps_ivar_dir, ivar_plot_fn)[:-5], p)

            log.info(f'{sv}, {m}, set{i} survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_set_mask)):.5f}')

            # save mask
            ivar_set_mask_fn = os.path.join(mask_dir, f'xtra_mask_{sv}_{m}_set{i}.fits')
            enmap.write_map(ivar_set_mask_fn, ivar_set_mask.astype(np.float32))

            # save plot of mask
            if save_plot_mask:
                p = enplot.plot(ivar_set_mask, downgrade=8, ticks=1, colorbar=True)
                enplot.write(ivar_set_mask_fn[:-5], p)

            # also build xtra masks that are the union and intersection
            # of all the xtra masks
            mask_intersect = np.logical_and(mask_intersect, ivar_set_mask)
            mask_union = np.logical_or(mask_union, ivar_set_mask)
            ivar_mask = np.logical_and(ivar_mask, ivar_set_mask)

        log.info(f'{sv}, {m} intersection survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_mask)):.5f}')

        # save mask
        ivar_mask_fn = os.path.join(mask_dir, f'xtra_mask_{sv}_{m}.fits')
        enmap.write_map(ivar_mask_fn, ivar_mask.astype(np.float32))

        # save plot of mask
        if save_plot_mask:
            p = enplot.plot(ivar_mask, downgrade=8, ticks=1, colorbar=True)
            enplot.write(ivar_mask_fn[:-5], p)

log.info(f'All intersection survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(mask_intersect)):.5f}')
log.info(f'All union survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(mask_union)):.5f}')

# plot and save union and intersect masks
p = enplot.plot(mask_intersect, downgrade=8, ticks=1, colorbar=True)
enplot.write(os.path.join(mask_dir, f'xtra_mask_intersect'), p)
enmap.write_map(os.path.join(mask_dir, f'xtra_mask_intersect.fits'), mask_intersect.astype(np.float32))\

p = enplot.plot(mask_union, downgrade=8, ticks=1, colorbar=True)
enplot.write(os.path.join(mask_dir, f'xtra_mask_union'), p)
enmap.write_map(os.path.join(mask_dir, f'xtra_mask_union.fits'), mask_union.astype(np.float32))
