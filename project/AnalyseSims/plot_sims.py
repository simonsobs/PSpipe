#!/usr/bin/env python

from __future__ import print_function
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir='window'
pspy_utils.create_directory(window_dir)
mcm_dir='mcm'
plot_dir='plot'

pspy_utils.create_directory(mcm_dir)
pspy_utils.create_directory(plot_dir)

experiment=d['experiment']

for exp in experiment:
    
    freqs=d['freq_%s'%exp]
    
    for cont in content:
        maps_list= d['%s_maps'%cont]
        for map,f in zip(maps_list,freqs):
            map=so_map.read_map(map)
            map.plot(file_name='%s/%s_%s_%s.png'%(plot_dir,cont,exp,f))


    for f in freqs:
        mask=so_map.read_map(d['mask_%s_%s'%(exp,f)])
        mask.plot(file_name='%s/mask_%s_%s.png'%(plot_direxp,f))

