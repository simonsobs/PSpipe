'''
This script is used to generate simulations of Planck data, add the ffp10 noise simulations and compute their power spectra.
to run it:
python planck_sim_spectra_ffp10.py global_ffp10.dict
'''


import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing,so_mpi
import os,sys
from pixell import enmap,curvedsky,powspec
import time
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

auxMapDir='window'
mcmDir='mcm'
ps_model_dir='model'
theoryFgDir='theory_and_fg'


spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

iStart=d['iStart']
iStop=d['iStop']

freqs=d['freqs']
niter=d['niter']
lmax=d['lmax']
type=d['type']
binning_file=d['binning_file']
remove_mono_dipo_T=d['remove_mono_dipo_T']
remove_mono_dipo_pol=d['remove_mono_dipo_pol']
experiment='Planck'
splits=d['splits']
include_foregrounds=d['include_foregrounds']

simSpectraDir='sim_spectra_ffp10'

pspy_utils.create_directory(simSpectraDir)

nside=2048
ncomp=3

template=so_map.healpix_template(ncomp,nside)


if include_foregrounds==True:
    ps_th=np.load('%s/signal_fg_matrix.npy'%theoryFgDir)
else:
    ps_th=powspec.read_spectrum(d['theoryfile'])[:ncomp,:ncomp]


nSplits=len(splits)

pixwin=hp.pixwin(nside)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d['iStart'], imax=d['iStop'])

for iii in subtasks:

    t0=time.time()
    alms={}

    sim_alm=curvedsky.rand_alm(ps_th, lmax=lmax-1) #the nlms and sim_alm lmax need to be investigated, there is a mismatch of 1 that is not understood at the moment

    for freq_id,freq in enumerate(freqs):
        
        maps=d['map_%s'%freq]
        
        if include_foregrounds==True:
            freq_alm=np.zeros((3,sim_alm.shape[1]),dtype='complex')
            freq_alm[0]=sim_alm[0+freq_id*3].copy()
            freq_alm[1]=sim_alm[1+freq_id*3].copy()
            freq_alm[2]=sim_alm[2+freq_id*3].copy()

        for hm,map,k in zip(splits,maps,np.arange(nSplits)):
            
            if include_foregrounds==True:
                my_alms=freq_alm.copy()
            else:
                my_alms=sim_alm.copy()
            
            
            l,bl_T= np.loadtxt(d['beam_%s_%s_T'%(freq,hm)],unpack=True)
            l,bl_pol= np.loadtxt(d['beam_%s_%s_pol'%(freq,hm)],unpack=True)

            my_alms[0]=hp.sphtfunc.almxfl(my_alms[0],bl_T)
            my_alms[1]=hp.sphtfunc.almxfl(my_alms[1],bl_pol)
            my_alms[2]=hp.sphtfunc.almxfl(my_alms[2],bl_pol)
            
            for i in range(3):
                my_alms[i]=hp.sphtfunc.almxfl(my_alms[i],pixwin)

            pl_map=sph_tools.alm2map(my_alms,template)
            noise_map=so_map.read_map('%s/%s/ffp10_noise_%s_%s_map_mc_%05d.fits'%(d['ffp10_dir'],freq,freq,hm,iii))
            noise_map.data*=10**6
            pl_map.data+=noise_map.data

            window_T=so_map.read_map('%s/window_T_%s_%s-%s.fits'%(auxMapDir,experiment,freq,hm))
            window_pol=so_map.read_map('%s/window_P_%s_%s-%s.fits'%(auxMapDir,experiment,freq,hm))
            window_tuple=(window_T,window_pol)
            del window_T,window_pol

            cov_map=so_map.read_map('%s'%map,fields_healpix=4)
            badpix = (cov_map.data==hp.pixelfunc.UNSEEN)
            for i in range(3):
                pl_map.data[i][badpix]= 0.0
            if remove_mono_dipo_T:
                pl_map.data[0]=planck_utils.subtract_mono_di(pl_map.data[0], window_tuple[0].data, pl_map.nside )
            if remove_mono_dipo_pol:
                pl_map.data[1]=planck_utils.subtract_mono_di(pl_map.data[1], window_tuple[1].data, pl_map.nside)
                pl_map.data[2]=planck_utils.subtract_mono_di(pl_map.data[2], window_tuple[1].data, pl_map.nside)

            alms[hm,freq]=sph_tools.get_alms(pl_map,window_tuple,niter,lmax)

    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

    Db_dict={}
    spec_name_list=[]
    for c1,freq1 in enumerate(freqs):
        for c2,freq2 in enumerate(freqs):
            if c1>c2: continue
            for s1,hm1 in enumerate(splits):
                for s2,hm2 in enumerate(splits):
                    if (s1>s2) & (c1==c2): continue
                
                    prefix= '%s/%s_%sx%s_%s-%sx%s'%(mcmDir,experiment,freq1,experiment,freq2,hm1,hm2)

                    mcm_inv,mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs,unbin=True)

                    l,ps= so_spectra.get_spectra(alms[hm1,freq1],alms[hm2,freq2],spectra=spectra)
                    spec_name='%s_%sx%s_%s-%sx%s'%(experiment,freq1,experiment,freq2,hm1,hm2)
                    l,cl,lb,Db=planck_utils.process_planck_spectra(l,ps,binning_file,lmax,type=type,mcm_inv=mcm_inv,spectra=spectra)
                    spec_name_list+=[spec_name]
                    so_spectra.write_ps('%s/sim_spectra_%s_%04d.dat'%(simSpectraDir,spec_name,iii),lb,Db,type=type,spectra=spectra)
                    so_spectra.write_ps('%s/sim_spectra_unbin_%s_%04d.dat'%(simSpectraDir,spec_name,iii),l,cl,type=type,spectra=spectra)

    print ('sim %04d take %.02f seconds to compute'%(iii,time.time()-t0))
