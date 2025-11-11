from pspy import so_dict

from pixell import enmap, enplot      

import numpy as np

import os
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# FIXME: won't fork for LF
mapname = d['arrays_so'][0]
footprint_geometry = enmap.read_map_geometry(d[f'maps_so_{mapname}'][0])

raw_act_map_dir = d['raw_act_maps_dir']
for mapname in d[f'arrays_dr6']:
    print(mapname)

    # loop over splits. for first split, get the coadd fn too
    for fn in d[f'maps_dr6_{mapname}']:
        act_map_dir, basename = os.path.split(fn)

        if 'set0' in basename:
            coadd_basename = basename.replace('set0', 'coadd')

        act_extracted = enmap.read_map(f'{raw_act_map_dir}/{basename}', geometry=footprint_geometry)
        enmap.write_map(f'{act_map_dir}/{basename}', act_extracted)

        # for ps maps
        if d['src_free_maps_dr6']:
            basename = basename.replace('_srcfree.fits', '.fits')
            act_extracted = enmap.read_map(f'{raw_act_map_dir}/{basename}', geometry=footprint_geometry)
            enmap.write_map(f'{act_map_dir}/{basename}', act_extracted)

        # ivar maps. assumes 'map_srcfree' if srcfree else 'map'
        basename = basename.replace('map', 'ivar')
        act_extracted = enmap.read_map(f'{raw_act_map_dir}/{basename}', geometry=footprint_geometry)
        enmap.write_map(f'{act_map_dir}/{basename}', act_extracted)

    # do the coadd fn
    act_extracted = enmap.read_map(f'{raw_act_map_dir}/{coadd_basename}', geometry=footprint_geometry)
    enmap.write_map(f'{act_map_dir}/{coadd_basename}', act_extracted)

    # for ps maps
    if d['src_free_maps_dr6']:
        coadd_basename = coadd_basename.replace('_srcfree.fits', '.fits')
        act_extracted = enmap.read_map(f'{raw_act_map_dir}/{coadd_basename}', geometry=footprint_geometry)
        enmap.write_map(f'{act_map_dir}/{coadd_basename}', act_extracted)

    # ivar maps. assumes 'map_srcfree' if srcfree else 'map'
    coadd_basename = coadd_basename.replace('map', 'ivar')
    act_extracted = enmap.read_map(f'{raw_act_map_dir}/{coadd_basename}', geometry=footprint_geometry)
    enmap.write_map(f'{act_map_dir}/{coadd_basename}', act_extracted)