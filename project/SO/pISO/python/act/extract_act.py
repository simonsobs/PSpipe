description="""Uses the lat iso map geometry to load the published dr6 maps,
extract them to the lat iso geometry, and save them into the paramfile's 
location for dr6 maps.
"""

from pspy import so_dict

from pixell import enmap      

import os
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('published-act-map-dir', type=str,
                    help='Path to published dr6 maps on this cluster')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
published_act_map_dir = args.published_act_map_dir

# FIXME: won't fork for LF
sv = 'lat_iso'
mapname = d[f'arrays_{sv}'][0]
footprint_geometry = enmap.read_map_geometry(d[f'maps_{sv}_{mapname}'][0])

for mapname in d[f'arrays_dr6']:
    print(mapname)

    # loop over splits. for first split, get the coadd fn too
    for fn in d[f'maps_dr6_{mapname}']:
        act_map_dir, basename = os.path.split(fn)

        if 'set0' in basename:
            coadd_basename = basename.replace('set0', 'coadd')

        act_extracted = enmap.read_map(f'{published_act_map_dir}/{basename}', geometry=footprint_geometry)
        enmap.write_map(f'{act_map_dir}/{basename}', act_extracted)

        # for ps maps
        if d['src_free_maps_dr6']:
            basename = basename.replace('_srcfree.fits', '.fits')
            act_extracted = enmap.read_map(f'{published_act_map_dir}/{basename}', geometry=footprint_geometry)
            enmap.write_map(f'{act_map_dir}/{basename}', act_extracted)

        # ivar maps. assumes 'map_srcfree' if srcfree else 'map'
        basename = basename.replace('map', 'ivar')
        act_extracted = enmap.read_map(f'{published_act_map_dir}/{basename}', geometry=footprint_geometry)
        enmap.write_map(f'{act_map_dir}/{basename}', act_extracted)

    # do the coadd fn
    act_extracted = enmap.read_map(f'{published_act_map_dir}/{coadd_basename}', geometry=footprint_geometry)
    enmap.write_map(f'{act_map_dir}/{coadd_basename}', act_extracted)

    # for ps maps
    if d['src_free_maps_dr6']:
        coadd_basename = coadd_basename.replace('_srcfree.fits', '.fits')
        act_extracted = enmap.read_map(f'{published_act_map_dir}/{coadd_basename}', geometry=footprint_geometry)
        enmap.write_map(f'{act_map_dir}/{coadd_basename}', act_extracted)

    # ivar maps. assumes 'map_srcfree' if srcfree else 'map'
    coadd_basename = coadd_basename.replace('map', 'ivar')
    act_extracted = enmap.read_map(f'{published_act_map_dir}/{coadd_basename}', geometry=footprint_geometry)
    enmap.write_map(f'{act_map_dir}/{coadd_basename}', act_extracted)