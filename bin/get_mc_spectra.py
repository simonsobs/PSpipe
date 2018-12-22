#!/usr/bin/env python

from __future__ import print_function
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict, so_mpi, so_misc
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

result_dir=d['result_dir']
window= so_map.read_map('%s/window.fits'%(result_dir))

if d['spin']=='0-2':
    ncomp=3
    spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']
    if window.pixel=='CAR':
        template=so_map.car_template(ncomp,d['ra0'],d['ra1'],d['dec0'],d['dec1'],d['res'])
    else:
        template=so_map.healpix_template(ncomp=ncomp,nside=d['nside'])
    window=(window,window)
    mbb_inv,Bbl=so_mcm.read_coupling(prefix='%s/test'%result_dir,spin_pairs=spin_pairs)

elif d['spin']=='0':
    ncomp=1
    spectra=None
    if window.pixel=='CAR':
        template=so_map.car_template(ncomp,d['ra0'],d['ra1'],d['dec0'],d['dec1'],d['res'])
    else:
        template=so_map.healpix_template(ncomp=ncomp,nside=d['nside'])
    mbb_inv,Bbl=so_mcm.read_coupling(prefix='%s/test'%result_dir)

# trigger mpi
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d['iStart'], imax=d['iStop'])
for iii in subtasks:
    print ('sim number %04d'%iii)
    cmb=template.synfast(d['clfile'])
    splitlist=[]
    nameList=[]
    for i in range(d['nSplits']):
        split=cmb.copy()
        noise = so_map.white_noise(split,rms_uKarcmin_T=d['rms_uKarcmin_T'])
        split.data+=noise.data
        splitlist+=[split]
        nameList+=['split_%d'%i]
    
    almList=[]
    for s in splitlist:
        almList+=[ sph_tools.get_alms(s,window,d['niter'],d['lmax']) ]

    spec_list = open('%s/spectra_list.txt'%(result_dir),mode="w")
    for name1, alm1, c1  in zip(nameList,almList,np.arange(d['nSplits'])):
        for name2, alm2, c2  in zip(nameList,almList,np.arange(d['nSplits'])):
            if c1>c2: continue
            ls,ps= so_spectra.get_spectra(alm1,alm2,spectra=spectra)
            spec_name='%sx%s'%(name1,name2)
            lb,Db=so_spectra.bin_spectra(ls,ps,d['binning_file'],d['lmax'],type=d['type'],mbb_inv=mbb_inv,spectra=spectra)
            so_spectra.write_ps('%s/spectra_ncomp%d_%s_%04d.dat'%(result_dir,ncomp,spec_name,iii),lb,Db,type=d['type'],spectra=spectra)
            spec_list.write('%s \n'%spec_name)
    spec_list.close()
            
 
