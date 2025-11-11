from pspy import so_dict

from pixell import enmap, enplot      

import numpy as np

import os
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

mask_dir = d['mask_dir']
save_plot = True
show_plot = False

# get reasonable ivar, top 90% of nonzero values seems to work decently
# TODO: make dynamic
ivar_smooth_deg = 0.2
ivar_quantile = 0.1

mask_intersect = True
mask_union = False
for sv in d['surveys']:
    for m in d[f'arrays_{sv}']:
        print(m)

        # get path in which the coadd map lives
        # (do it this way in case the paramfile lists the split maps for maps_so,
        # for example. this way we unambigously load the coadd map, if it exists.)
        map_dir, map_fn = os.path.split(d[f'maps_{sv}_{m}'][0])
        if d[f"src_free_maps_{sv}"] == True:
            ivar_coadd_fn = map_fn.replace('map_srcfree', 'ivar')
        else:
            ivar_coadd_fn = map_fn.replace('map', 'ivar')
        ivar_coadd_fn = ivar_coadd_fn.replace('set0', 'coadd')

        # mask is based on the smoothed coadd ivar map
        # only the pixels were the original ivar were nonzero though
        ivar = enmap.read_map(f'{map_dir}/{ivar_coadd_fn}')  
        ivar_smooth = enmap.smooth_gauss(ivar, np.deg2rad(ivar_smooth_deg))
        ivar_mask = ivar_smooth > np.quantile(ivar_smooth[ivar > 0], ivar_quantile)
        
        # build xtra masks that are the union and intersection of all the xtra masks
        mask_intersect = np.logical_and(mask_intersect, ivar_mask)
        mask_union = np.logical_or(mask_union, ivar_mask)
        
        # save map
        enmap.write_map(f'{mask_dir}/xtra_mask_{sv}_{m}.fits', ivar_mask.astype(np.float32))

        # save plot of map
        if save_plot:
            p = enplot.plot(ivar_mask, downgrade=8, ticks=1, colorbar=True)
            enplot.write(f'{mask_dir}/xtra_mask_{sv}_{m}', p)

        # optionally show plot of map
        if show_plot:
            enplot.show(p)
            
# save union and intersect
enmap.write_map(f'{mask_dir}/xtra_mask_intersect.fits', mask_intersect.astype(np.float32))
enmap.write_map(f'{mask_dir}/xtra_mask_union.fits', mask_union.astype(np.float32))

# save plot of union and intersect
if save_plot:
    p = enplot.plot([mask_intersect, mask_union], downgrade=8, colorbar=True, ticks=1)
    enplot.write(f'{mask_dir}/xtra_mask_intersect_and_union', p)

# optionally show plot of union and intersect
if show_plot:
    enplot.show(p)