import numpy as np
from pixell import enmap
from os.path import join as opj
from pspy import so_mpi
from itertools import product

map_dir = '/home/zatkins/scratch/projects/lat-iso/piso/maps/lat/wide'
map_fn_template = '{t}_all_{tube}_4way_0{split}_{freq}_sky_{maptype}.fits'
coadd_fn_template = '{t}_all_{tube}_4way_coadd_{freq}_sky_{maptype}.fits'

types = ['type1', 'type3']
maps = ['i1_f090', 'i1_f150', 'i3_f090', 'i3_f150', 'i4_f090', 'i4_f150', 'i6_f090', 'i6_f150',
        'c1_f220', 'c1_f280', 'i5_f220', 'i5_f280']
type_maps = list(product(types, maps))

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(type_maps) - 1)

for i in subtasks:
    t, m = type_maps[i]
    print(t, m)
    tube, freq = m.split('_')
    coadd_ivar = 0
    coadd_map = 0
    for i in range(4):
        ivar = enmap.read_map(opj(map_dir, map_fn_template.format(t=t, tube=tube, split=i, freq=freq, maptype='ivar')))
        imap = enmap.read_map(opj(map_dir, map_fn_template.format(t=t, tube=tube, split=i, freq=freq, maptype='map0050')))

        coadd_ivar += ivar 
        coadd_map += ivar*imap

    coadd_map = np.divide(coadd_map, coadd_ivar, where=coadd_ivar>0, out=0*coadd_map)

    enmap.write_map(opj(map_dir, coadd_fn_template.format(t=t, tube=tube, freq=freq, maptype='ivar')), coadd_ivar)
    enmap.write_map(opj(map_dir, coadd_fn_template.format(t=t, tube=tube, freq=freq, maptype='map0050')), coadd_map)