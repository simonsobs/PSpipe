#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing,so_mpi
import os,sys
from pixell import enmap
import time,os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectraDir='spectra'
mcmDir='mcm'

pspy_utils.create_directory(spectraDir)
pspy_utils.create_directory(mcmDir)



spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

arrays=d['arrays']
niter=d['niter']
lmax=3000
type=d['type']
binning_file=d['binning_file']
theoryfile=d['theoryfile']
fsky={}
fsky['pa1']='fsky0.01081284'
fsky['pa2']='fsky0.01071187'

apo = so_map.read_map(d['apo_path'])
box=so_map.bounding_box_from_map(apo)
recompute_mcm=False
clfile=d['theoryfile']


for ar in ['pa1']:
    t=time.time()

    window=so_map.read_map(d['window_T_%s'%ar])
    window=so_map.get_submap_car(window,box,mode='round')
    
    window_tuple=(window,window)
    
    print ("compute mcm and Bbl ...")
    beam= np.loadtxt(d['beam_%s'%ar])
    l,bl=beam[:,0],beam[:,1]
    bl_tuple=(bl,bl)
    
    if recompute_mcm==True:
        mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(window_tuple, binning_file,niter=niter, bl1=bl_tuple, lmax=lmax, type=type,save_file='%s/%s'%(mcmDir,ar))
    else:
        spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']
        mbb_inv,Bbl=so_mcm.read_coupling(prefix='%s/%s'%(mcmDir,ar),spin_pairs=spin_pairs)

    almList=[]
    nameList=[]
    
    map_T=d['map_T_%s'%ar][0]
    map_Q=d['map_Q_%s'%ar][0]
    map_U=d['map_U_%s'%ar][0]

    print ("compute harmonic transform ...")


    so_mpi.init(True)
    subtasks = so_mpi.taskrange(imin=d['iStart'], imax=d['iStop'])

    for iii in range(subtasks):
        t0=time.time()
        print (iii)
  
        template=so_map.from_components(map_T,map_Q,map_U)
        template=so_map.get_submap_car(template,box,mode='floor')
        
        cmb_car=template.synfast(clfile)
        noise0 = so_map.white_noise(template,rms_uKarcmin_T=15)
        noise1 = so_map.white_noise(template,rms_uKarcmin_T=15)
        
        split0=cmb_car.copy()
        split0.data+=noise0.data
        split1=cmb_car.copy()
        split1.data+=noise1.data
        
        split0_filt=split0.copy()
        split1_filt=split1.copy()

        split0_filt = so_map_preprocessing.get_map_kx_ky_filtered_pyfftw(split0_filt,apo,d['filter_dict'])
        split1_filt = so_map_preprocessing.get_map_kx_ky_filtered_pyfftw(split1_filt,apo,d['filter_dict'])

        alm0= sph_tools.get_alms(split0,window_tuple,niter,lmax)
        alm1= sph_tools.get_alms(split1,window_tuple,niter,lmax)

        alm0_filt= sph_tools.get_alms(split0_filt,window_tuple,niter,lmax)
        alm1_filt= sph_tools.get_alms(split1_filt,window_tuple,niter,lmax)

        l,ps= so_spectra.get_spectra(alm0,alm1,spectra=spectra)
        l,ps_filt= so_spectra.get_spectra(alm0_filt,alm1_filt,spectra=spectra)

        lb,Db_dict=so_spectra.bin_spectra(l,ps,binning_file,lmax,type=type,mbb_inv=mbb_inv,spectra=spectra)
        lb,Db_dict_filt=so_spectra.bin_spectra(l,ps_filt,binning_file,lmax,type=type,mbb_inv=mbb_inv,spectra=spectra)

        so_spectra.write_ps('%s/spectra_%03d.dat'%(spectraDir,iii),lb,Db_dict,type=type,spectra=spectra)
        so_spectra.write_ps('%s/spectra_filt_%03d.dat'%(spectraDir,iii),lb,Db_dict_filt,type=type,spectra=spectra)
        print (time.time()-t0)
