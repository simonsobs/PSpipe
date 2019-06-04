import matplotlib
matplotlib.use('Agg')
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

survey_dir='survey_mask'

pspy_utils.create_directory(survey_dir)

experiment=d['experiment']

for exp in experiment:
    freqs=d['freq_%s'%exp]
    maps_list= d['noise_maps']
    for map,f in zip(maps_list,freqs):
        mask=so_map.read_map(map)
        id=np.where(mask.data!=0)
        mask.data[id]==1
        mask.write_map('%s/survey_mask_%s_%s.fits'%(survey_dir,exp,f))
        mask.plot(file_name='%s/mask_%s_%s'%(plot_dir,exp,f))
