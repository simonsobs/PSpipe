'''
This script is used to generate simulations of Planck data and compute their power spectra.
to run it:
python planck_sim_spectra.py global.dict
Note that we are using homogeneous non white noise here
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
include_sys=d['include_systematics']
include_foregrounds=d['include_foregrounds']
use_noise_th=d['use_noise_th']

if include_sys==True:
    simSpectraDir='sim_spectra_syst'
else:
    simSpectraDir='sim_spectra'

pspy_utils.create_directory(simSpectraDir)

nside=2048
ncomp=3

template=so_map.healpix_template(ncomp,nside)


if include_foregrounds==True:
    ps_th=np.load('%s/signal_fg_matrix.npy'%theoryFgDir)
else:
    ps_th=powspec.read_spectrum(d['theoryfile'])[:ncomp,:ncomp]


nSplits=len(splits)

l,Nl_T,Nl_P=planck_utils.get_noise_matrix_spin0and2(ps_model_dir,experiment,freqs,lmax,nSplits,lcut=0,use_noise_th=use_noise_th)

pixwin=hp.pixwin(nside)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d['iStart'], imax=d['iStop'])

for iii in subtasks:

    t0=time.time()
    alms={}

    sim_alm=curvedsky.rand_alm(ps_th, lmax=lmax-1) #the nlms and sim_alm lmax need to be investigated, there is a mismatch of 1 that is not understood at the moment
    nlms=planck_utils.generate_noise_alms(Nl_T,Nl_P,lmax,nSplits,ncomp)

    for freq_id,freq in enumerate(freqs):
        
        
        maps=d['map_%s'%freq]
        
        if include_foregrounds==True:
            freq_alm=np.zeros((3,sim_alm.shape[1]),dtype='complex')
            freq_alm[0]=sim_alm[0+freq_id*3].copy()
            freq_alm[1]=sim_alm[1+freq_id*3].copy()
            freq_alm[2]=sim_alm[2+freq_id*3].copy()

        for hm,map,k in zip(splits,maps,np.arange(nSplits)):
            
            if include_foregrounds==True:
                noisy_alms=freq_alm.copy()
            else:
                noisy_alms=sim_alm.copy()
            
            if include_sys==True:
                l,bl_T= np.loadtxt(d['beam_%s_%s_T_syst'%(freq,hm)],unpack=True)
                l,bl_pol= np.loadtxt(d['beam_%s_%s_pol_syst'%(freq,hm)],unpack=True)
            else:
                l,bl_T= np.loadtxt(d['beam_%s_%s_T'%(freq,hm)],unpack=True)
                l,bl_pol= np.loadtxt(d['beam_%s_%s_pol'%(freq,hm)],unpack=True)

            noisy_alms[0]=hp.sphtfunc.almxfl(noisy_alms[0],bl_T)
            noisy_alms[1]=hp.sphtfunc.almxfl(noisy_alms[1],bl_pol)
            noisy_alms[2]=hp.sphtfunc.almxfl(noisy_alms[2],bl_pol)

            noisy_alms[0] +=  nlms['T',k][freq_id]
            noisy_alms[1] +=  nlms['E',k][freq_id]
            noisy_alms[2] +=  nlms['B',k][freq_id]
            
            if include_sys==True:
                l,Tl_T=np.loadtxt(d['TF_%s_%s_T'%(freq,hm)],unpack=True)
                l,Tl_pol=np.loadtxt(d['TF_%s_%s_pol'%(freq,hm)],unpack=True)
            
                noisy_alms[0]=hp.sphtfunc.almxfl(noisy_alms[0],Tl_T)
                noisy_alms[1]=hp.sphtfunc.almxfl(noisy_alms[1],Tl_pol)
                noisy_alms[2]=hp.sphtfunc.almxfl(noisy_alms[2],Tl_pol)

            
            for i in range(3):
                noisy_alms[i]=hp.sphtfunc.almxfl(noisy_alms[i],pixwin)

            pl_map=sph_tools.alm2map(noisy_alms,template)

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
