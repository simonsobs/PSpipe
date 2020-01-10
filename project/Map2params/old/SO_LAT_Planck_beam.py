#!/usr/bin/env python

from pspy import  pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


experiment=d['experiment']

pspy_utils.create_directory('beam')

beam_FWHM={}
beam_FWHM['LAT27']=7.4
beam_FWHM['LAT39']=5.1
beam_FWHM['LAT93']=2.2
beam_FWHM['LAT145']=1.4
beam_FWHM['LAT225']=1.0
beam_FWHM['LAT280']=0.9

beam_FWHM['Planck100']=9.68
beam_FWHM['Planck143']= 7.30
beam_FWHM['Planck217']=5.02
beam_FWHM['Plancks353']=4.94

for exp in experiment:
    freqs=d['freq_%s'%exp]
    for f in freqs:
        beam_FWHM_rad = np.deg2rad(beam_FWHM[exp+f])/60
        beam = beam_FWHM_rad/np.sqrt(8*np.log(2))
        l=np.arange(2,10000)
        bl=np.exp(-l*(l+1)*beam**2/2.)
        np.savetxt('beam/beam_%s_%s.dat'%(exp,f), np.transpose([l,bl]))

