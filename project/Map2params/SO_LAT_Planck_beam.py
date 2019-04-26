#!/usr/bin/env python

from pspy import  pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

freqs=d['freqs']

pspy_utils.create_directory('beam')

beam_FWHM={}
beam_FWHM['LAT_27']=7.4
beam_FWHM['LAT_39']=5.1
beam_FWHM['LAT_93']=2.2
beam_FWHM['LAT_145']=1.4
beam_FWHM['LAT_225']=1.0
beam_FWHM['LAT_280']=0.9

beam_FWHM['Planck_100']=9.68
beam_FWHM['Planck_143']= 7.30
beam_FWHM['Planck_217']=5.02
beam_FWHM['Planck_353']=4.94

for f in freqs:
    beam_FWHM_rad = np.deg2rad(beam_FWHM[f])/60
    beam = beam_FWHM_rad/np.sqrt(8*np.log(2))
    l=np.arange(2,10000)
    bl=np.exp(-l*(l+1)*beam**2/2.)
    np.savetxt('beam/beam_%s.dat'%f, np.transpose([l,bl]))

