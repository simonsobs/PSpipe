from pspy import so_dict

from pixell import enmap, enplot      

import numpy as np

import os
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

mask_dir = d['mask_dir']

mask_intersect = True
mask_union = False

plot = False

# get reasonable ivar, top 90% of nonzero values seems to work decently
# TODO: make dynamic
ivar_quantile = 0.1

for s in d['arrays_so']:
    print(s)

    # get path in which the coadd map lives
    # (do it this way in case the paramfile lists the split maps for maps_so,
    # for example. this way we unambigously load the coadd map, if it exists.)
    map_dir = os.path.dirname(d[f'maps_so_{s}'])

    # mask is based on the coadd ivar map
    ivar = enmap.read_map(f'{map_dir}/sky_div_{s}_20250801.fits')  
    ivar_mask = ivar > np.quantile(ivar[ivar > 0], ivar_quantile)
    
    # build xtra masks that are the union and intersection of all the xtra masks
    mask_intersect = np.logical_and(mask_intersect, ivar_mask)
    mask_union = np.logical_or(mask_union, ivar_mask)
    
    # save map
    enmap.write_map(f'{mask_dir}/xtra_mask_{s}_20250801.fits', ivar_mask.astype(np.float32))

    # save plot of map
    p = enplot.plot(ivar_mask, downgrade=8, ticks=1, colorbar=True)
    enplot.write(f'{mask_dir}/xtra_mask_{s}_20250801', p)

    # optionally show plot of map
    if plot:
        enplot.show(p)
        imap = enmap.read_map(f'{map_dir}/sky_map_{s}_20250801.fits', sel=np.s_[..., 0, :, :])
        enplot.pshow([imap, imap * ivar_mask], downgrade=8, colorbar=True, ticks=1, range=.000500)

# save union and intersect
enmap.write_map(f'{mask_dir}/xtra_mask_intersect_20250801.fits', mask_intersect.astype(np.float32))
enmap.write_map(f'{mask_dir}/xtra_mask_union_20250801.fits', mask_union.astype(np.float32))

# save plot of union and intersect
p = enplot.plot([mask_intersect, mask_union], downgrade=8, colorbar=True, ticks=1)
enplot.write(f'{mask_dir}/xtra_mask_intersect_and_union_20250801', p)

# optionally show plot of union and intersect
if plot:
    enplot.show(p)