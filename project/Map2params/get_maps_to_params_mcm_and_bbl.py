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

experiment=d['experiment']
lmax=d['lmax']
ncomp=3
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']


for exp in experiment:
    
    freqs=d['freq_%s'%exp]

    if d['pixel_%s'%exp]=='CAR':
        binary=so_map.car_template(1,d['ra0_%s'%exp],d['ra1_%s'%exp],d['dec0_%s'%exp],d['dec1_%s'%exp],d['res_%s'%exp])
        binary.data[:]=0
        binary.data[1:-1,1:-1]=1

    elif d['pixel_%s'%exp]=='HEALPIX':
        binary=so_map.healpix_template(ncomp=1,nside=d['nside_%s'%exp])
        binary.data[:]=1
        if d['disc_%s'%exp]==True:
            binary.data[:]=0
            vec=hp.pixelfunc.ang2vec(d['lon_disc_%s'%exp],d['lat_disc_%s'%exp], lonlat=True)
            disc=hp.query_disc(d['nside_%s'%exp], vec, radius=d['radius_disc_%s'%exp]*np.pi/180)
            binary.data[disc]=1

    freq=d['freq_%s'%exp]

    for f in freqs:
        window=binary.copy()
        if d['galactic_mask_%s'%exp]==True:
            gal_mask=so_map.read_map(d['galactic_mask_%s_file_%s'%(exp,f)])
            gal_mask.plot(file_name='%s/gal_mask_%s_%s'%(window_dir,exp,f))
            window.data[:]*=gal_mask.data[:]
        if d['survey_mask_%s'%exp]==True:
            survey_mask=so_map.read_map(d['survey_mask_%s_file_%s'%(exp,f)])
            survey_mask.plot(file_name='%s/survey_mask_mask_%s_%s'%(window_dir,exp,f))
            window.data[:]*=survey_mask.data[:]

        apo_radius_degree=(d['apo_radius_survey_%s'%exp])#* np.random.randint(50,200)/100
        window=so_window.create_apodization(window, apo_type=d['apo_type_survey_%s'%exp], apo_radius_degree=apo_radius_degree)

        if d['pts_source_mask_%s'%exp]==True:
            hole_radius_arcmin=(d['source_mask_radius_%s'%exp])#* np.random.randint(50,200)/100
            mask=so_map.simulate_source_mask(binary, nholes=d['source_mask_nholes_%s'%exp], hole_radius_arcmin=hole_radius_arcmin)
            mask= so_window.create_apodization(mask, apo_type=d['apo_type_mask_%s'%exp], apo_radius_degree=d['apo_radius_mask_%s'%exp])
            window.data*=mask.data

        window.write_map('%s/window_%s_%s.fits'%(window_dir,exp,f))
        window.plot(file_name='%s/window_%s_%s'%(window_dir,exp,f))


for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        l,bl1= np.loadtxt('beam/beam_%s_%s.dat'%(exp1,f1),unpack=True)
        window1=so_map.read_map('%s/window_%s_%s.fits'%(window_dir,exp1,f1))
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue

                l,bl2= np.loadtxt('beam/beam_%s_%s.dat'%(exp2,f2),unpack=True)
                window2=so_map.read_map('%s/window_%s_%s.fits'%(window_dir,exp2,f2))
                print (exp1,f1,exp2,f2)
                mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(win1=(window1,window1),win2=(window2,window2),bl1=(bl1,bl1),bl2=(bl2,bl2),binning_file= d['binning_file'],niter=d['niter'], lmax=d['lmax'], type=d['type'],save_file='%s/%s_%sx%s_%s'%(mcm_dir,exp1,f1,exp2,f2))



