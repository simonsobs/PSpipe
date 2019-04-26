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
pspy_utils.create_directory(mcm_dir)

freqs=d['freqs']
lmax=d['lmax']

if d['spin']=='0-2':
    ncomp=3
    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']
elif d['spin']=='0':
    ncomp=1

if d['pixel']=='CAR':
    print (d['ra0'],d['ra1'],d['dec0'],d['dec1'],d['res'])
    binary=so_map.car_template(1,d['ra0'],d['ra1'],d['dec0'],d['dec1'],d['res'])
    binary.data[:]=0
    binary.data[1:-1,1:-1]=1

elif d['pixel']=='HEALPIX':
    binary=so_map.healpix_template(ncomp=1,nside=d['nside'])
    vec=hp.pixelfunc.ang2vec(d['lon'],d['lat'], lonlat=True)
    disc=hp.query_disc(d['nside'], vec, radius=d['radius']*np.pi/180)
    binary.data[disc]=1

for f in freqs:
    apo_radius_degree=(d['apo_radius_survey'])#* np.random.randint(50,200)/100
    hole_radius_arcmin=(d['source_mask_radius'])#* np.random.randint(50,200)/100
    window=so_window.create_apodization(binary, apo_type=d['apo_type_survey'], apo_radius_degree=apo_radius_degree)
    mask=so_map.simulate_source_mask(binary, nholes=d['source_mask_nholes'], hole_radius_arcmin=hole_radius_arcmin)
    mask= so_window.create_apodization(mask, apo_type=d['apo_type_mask'], apo_radius_degree=d['apo_radius_mask'])
    window.data*=mask.data
    window.write_map('%s/window_%s.fits'%(window_dir,f))
#window.plot()

for c1,f1 in enumerate(freqs):
    l,bl1= np.loadtxt('beam/beam_%s.dat'%f1,unpack=True)
    window_1=so_map.read_map('%s/window_%s.fits'%(window_dir,f1))
    for c2,f2 in enumerate(freqs):
        if c1>c2 : continue
        l,bl2= np.loadtxt('beam/beam_%s.dat'%f2,unpack=True)
        window_2=so_map.read_map('%s/window_%s.fits'%(window_dir,f2))

        print (f1,f2)
        if ncomp==3:
            try:
                mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(win1=(window_1,window_1),win2=(window_2,window_2),bl1=bl1,bl2=bl2,binning_file= d['binning_file'],niter=0, lmax=d['lmax'], type=d['type'],save_file='%s/%s_%s'%(mcm_dir,f1,f2))
            except:
                plt.plot(bl1)
                plt.plot(bl2)
                plt.show()
                    # window_1.plot()
                    # window_2.plot()
        if ncomp==1:
            mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0(win1=window_1,win2=window_2,bl1=bl1,bl2=bl2,binning_file= d['binning_file'],niter=0, lmax=d['lmax'], type=d['type'],save_file='%s/%s_%s'%(mcm_dir,f1,f2))


