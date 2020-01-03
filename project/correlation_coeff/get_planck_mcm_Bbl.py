'''
This script is used to compute the mode coupling matrices of the Planck data.
The inputs for the script are the Planck beam and likelihood masks.
To run it:
python get_planck_mcm_Bbl.py global.dict
'''
import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

auxMapDir='window'
mcmDir='mcm'

pspy_utils.create_directory(auxMapDir)
pspy_utils.create_directory(mcmDir)

freqs=d['freqs']
niter=d['niter']
lmax=d['lmax']
type=d['type']
binning_file=d['binning_file']
pixWin=d['pixWin']
splits=d['splits']
experiment='Planck'

print ('Compute Planck 2018 mode coupling matrices')

for c1,freq1 in enumerate(freqs):
    
    window_T_1=d['window_T_%s'%freq1]
    window_pol_1=d['window_pol_%s'%freq1]

    for count1,hm1 in enumerate(splits):
        
        win_T1=so_map.read_map(window_T_1[count1])
        win_pol1=so_map.read_map(window_pol_1[count1])
        
        win_T1.write_map('%s/window_T_%s_%s-%s.fits'%(auxMapDir,experiment,freq1,hm1))
        win_pol1.write_map('%s/window_P_%s_%s-%s.fits'%(auxMapDir,experiment,freq1,hm1))

        window_tuple1=(win_T1,win_pol1)
        
        del win_T1,win_pol1
    
        l,bl1_T= np.loadtxt(d['beam_%s_%s_T'%(freq1,hm1)],unpack=True)
        l,bl1_pol=np.loadtxt(d['beam_%s_%s_pol'%(freq1,hm1)],unpack=True)

        if pixWin==True:
            bl1_T*=hp.pixwin(window_tuple1[0].nside)[:len(bl1_T)]
            bl1_pol*=hp.pixwin(window_tuple1[0].nside)[:len(bl1_pol)]
        
        bl_tuple1=(bl1_T,bl1_pol)

        for c2,freq2 in enumerate(freqs):
            if c1>c2: continue
            
            window_T_2=d['window_T_%s'%freq2]
            window_pol_2=d['window_pol_%s'%freq2]

            for count2,hm2 in enumerate(splits):
                if (count1>count2) & (c1==c2): continue
                print (freq1,freq2)

                win_T2=so_map.read_map(window_T_2[count2])
                win_pol2=so_map.read_map(window_pol_2[count2])

                window_tuple2=(win_T2,win_pol2)

                del win_T2,win_pol2
                
                l,bl2_T= np.loadtxt(d['beam_%s_%s_T'%(freq2,hm2)],unpack=True)
                l,bl2_pol=np.loadtxt(d['beam_%s_%s_pol'%(freq2,hm2)],unpack=True)

                if pixWin==True:
                    bl2_T*=hp.pixwin(window_tuple2[0].nside)[:len(bl2_T)]
                    bl2_pol*=hp.pixwin(window_tuple2[0].nside)[:len(bl2_pol)]

                bl_tuple2=(bl2_T,bl2_pol)
                
                mcm_inv,mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(win1=window_tuple1,win2=window_tuple2, binning_file=binning_file, bl1=bl_tuple1,bl2=bl_tuple2, lmax=lmax,niter=niter, type=type, unbin=True,save_file='%s/%s_%sx%s_%s-%sx%s'%(mcmDir,experiment,freq1,experiment,freq2,hm1,hm2),lmax_pad=d['lmax_pad'])




