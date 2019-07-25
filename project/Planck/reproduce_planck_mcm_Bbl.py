#import matplotlib
#matplotlib.use('Agg')
import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


auxMapDir='window'
mcmDir='mcm'

try:
    os.makedirs(mcmDir)
except:
    pass
try:
    os.makedirs(auxMapDir)
except:
    pass


arrays=d['arrays']
niter=d['niter']
lmax=d['lmax']
type=d['type']
binning_file=d['binning_file']
pixWin=d['pixWin']



print ('compute all Planck mode coupling matrices')

experiment='Planck'
split=['hm1','hm2']

for c1,ar1 in enumerate(arrays):
    maps1= d['map_%s'%ar1]
    for hm1,map1,count1 in zip(split,maps1,np.arange(2)):
        
        window_T_1=so_map.read_map(d['window_T_%s'%ar1])
        window_pol_1=so_map.read_map(d['window_pol_%s'%ar1])
        
        cov_map=so_map.read_map('%s'%map1,fields_healpix=4)
        badpix = (cov_map.data==hp.pixelfunc.UNSEEN)
        window_T_1.data[badpix]=0.0
        window_pol_1.data[badpix]=0.0
    
        window_T_1.write_map('%s/window_T_%s_%s-%s.fits'%(auxMapDir,experiment,ar1,hm1))
        window_pol_1.write_map('%s/window_P_%s_%s-%s.fits'%(auxMapDir,experiment,ar1,hm1))
        
        window_tuple1=(window_T_1,window_pol_1)
        
        del window_T_1,window_pol_1,cov_map
    
        beam1= np.loadtxt(d['beam_%s'%ar1])
        l,bl1=beam1[:,0],beam1[:,1]
        if pixWin==True:
            bl1*=hp.pixwin(window_tuple1[0].nside)[:len(bl1)]
    
        bl_tuple1=(bl1,bl1)

        for c2,ar2 in enumerate(arrays):
            if c1>c2: continue
            maps2= d['map_%s'%ar2]
            for hm2,map2,count2 in zip(split,maps2,np.arange(2)):
                print (count1,count2)
                if (count1>count2) & (c1==c2): continue

                window_T_2=so_map.read_map(d['window_T_%s'%ar2])
                window_pol_2=so_map.read_map(d['window_pol_%s'%ar2])

                cov_map=so_map.read_map('%s'%map2,fields_healpix=4)
                badpix = (cov_map.data==hp.pixelfunc.UNSEEN)
                window_T_2.data[badpix]=0.0
                window_pol_2.data[badpix]=0.0

                window_tuple2=(window_T_2,window_pol_2)

                del window_T_2,window_pol_2,cov_map

                beam2= np.loadtxt(d['beam_%s'%ar2])
                l,bl2=beam2[:,0],beam2[:,1]
                if pixWin==True:
                    bl2*=hp.pixwin(window_tuple2[0].nside)[:len(bl2)]
        
                bl_tuple2=(bl2,bl2)
                
                mcm_inv,mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(win1=window_tuple1,win2=window_tuple2, binning_file=binning_file, bl1=bl_tuple1,bl2=bl_tuple2, lmax=lmax,niter=niter, type=type, unbin=True,save_file='%s/%s_%sx%s_%s-%sx%s'%(mcmDir,experiment,ar1,experiment,ar2,hm1,hm2),lmax_pad=d['lmax_pad'])




