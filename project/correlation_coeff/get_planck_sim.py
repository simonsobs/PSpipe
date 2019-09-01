"""
This script is used to download the public planck data
To run it: python get_planck_data.py global.dict
It will download maps, likelihood masks and beams of planck
It will also generate a binning file suitable for the estimation of the cross correlation coefficient"
"""

import numpy as np,healpy as hp,pylab as plt
from pspy import pspy_utils,so_dict
import os,sys
import wget
import tarfile
import astropy.io.fits as fits

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# You have to spefify the data directory in which the products will be downloaded
data_dir=d['data_dir']
freqs = d['freqs']
splits = ['hm1','hm2' ]

print ('Download Planck simulation maps')
sim_dir=data_dir+'/sims'
pspy_utils.create_directory(sim_dir)
splits = ['hm1','hm2' ]

for iii in range(300):
    for hm in splits:
        for f in freqs:
            url='http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_%s_%s_map_mc_%05d.fits'%(f,hm,iii)
            print (url)
            wget.download(url, '%s/ffp10_noise_%s_%s_map_mc_%05d.fits'%(sim_dir,f,hm,iii))
