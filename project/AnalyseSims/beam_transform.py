#!/usr/bin/env python

from pspy import  pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


experiment=d['experiment']

pspy_utils.create_directory('beam')


beam_FWHM={}
beam_FWHM['la027']=7.4
beam_FWHM['la039']=5.1
beam_FWHM['la093']=2.2
beam_FWHM['la145']=1.4
beam_FWHM['la225']=1.0
beam_FWHM['la280']=0.9

for exp in experiment:
    freqs=d['freq_%s'%exp]
    for f in freqs:
        beam_FWHM_rad = np.deg2rad(beam_FWHM[exp+f])/60
        beam = beam_FWHM_rad/np.sqrt(8*np.log(2))
        l=np.arange(0,10000)
        bl=np.exp(-l*(l+1)*beam**2/2.)
        np.savetxt('beam/beam_%s_%s.dat'%(exp,f), np.transpose([l,bl]))

