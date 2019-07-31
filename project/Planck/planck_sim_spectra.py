'''
Note that we are using homogeneous non white noise here
'''

#import matplotlib
#matplotlib.use('Agg')
import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap,curvedsky,powspec
import time
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

auxMapDir='window'
mcmDir='mcm'
simSpectraDir='sim_spectra'
ps_model_dir='model'

try:
    os.makedirs(simSpectraDir)
except:
    pass

spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

iStart=d['iStart']
iStop=d['iStop']

arrays=d['arrays']
niter=d['niter']
lmax=d['lmax']
type=d['type']
binning_file=d['binning_file']
remove_mono_dipo_T=d['remove_mono_dipo_T']
remove_mono_dipo_pol=d['remove_mono_dipo_pol']
experiment='Planck'
split=['hm1','hm2']

nside=2048
ncomp=3

template=so_map.healpix_template(ncomp,nside)
ps_th=powspec.read_spectrum(d['theoryfile'])[:ncomp,:ncomp]

nSplits=len(split)
l,Nl_T,Nl_P=planck_utils.get_noise_matrix_spin0and2(ps_model_dir,experiment,arrays,lmax,nSplits,lcut=0)
pixwin=hp.pixwin(nside)

for iii in range(iStart,iStop):

    t0=time.time()
    alms={}

    sim_alm=curvedsky.rand_alm(ps_th, lmax=lmax-1) #the nlms and sim_alm lmax need to be investigated, there is a mismatch of 1 that is not understood at the moment
    nlms=planck_utils.generate_noise_alms(Nl_T,Nl_P,lmax,nSplits,ncomp)

    for arid,ar in  enumerate(arrays):
        maps=d['map_%s'%ar]
        alms_convolved=sim_alm.copy()
        
        beam= np.loadtxt(d['beam_%s'%ar])
        l,bl=beam[:,0],beam[:,1]

        alms_convolved=planck_utils.convolved_alms(alms_convolved,bl,ncomp)

        for hm,map,k in zip(split,maps,np.arange(nSplits)):
            
            noisy_alms=alms_convolved.copy()
                
            noisy_alms[0] +=  nlms['T',k][arid]
            noisy_alms[1] +=  nlms['E',k][arid]
            noisy_alms[2] +=  nlms['B',k][arid]
            
            noisy_alms=planck_utils.convolved_alms(noisy_alms,pixwin,ncomp)
                
            pl_map=sph_tools.alm2map(noisy_alms,template)

            window_T=so_map.read_map('%s/window_T_%s_%s-%s.fits'%(auxMapDir,experiment,ar,hm))
            window_pol=so_map.read_map('%s/window_P_%s_%s-%s.fits'%(auxMapDir,experiment,ar,hm))
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

            alms[hm,ar]=sph_tools.get_alms(pl_map,window_tuple,niter,lmax)

    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

    Db_dict={}
    spec_name_list=[]
    for c1,ar1 in enumerate(arrays):
        for c2,ar2 in enumerate(arrays):
            if c1>c2: continue
            for s1,hm1 in enumerate(split):
                for s2,hm2 in enumerate(split):
                    if (s1>s2) & (c1==c2): continue
                
                    prefix= '%s/%s_%sx%s_%s-%sx%s'%(mcmDir,experiment,ar1,experiment,ar2,hm1,hm2)

                    mcm_inv,mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs,unbin=True)

                    l,ps= so_spectra.get_spectra(alms[hm1,ar1],alms[hm2,ar2],spectra=spectra)
                    spec_name='%s_%sx%s_%s-%sx%s'%(experiment,ar1,experiment,ar2,hm1,hm2)
                    l,cl,lb,Db=planck_utils.process_planck_spectra(l,ps,binning_file,lmax,type=type,mcm_inv=mcm_inv,spectra=spectra)
                    spec_name_list+=[spec_name]
                    so_spectra.write_ps('%s/sim_spectra_%s_%04d.dat'%(simSpectraDir,spec_name,iii),lb,Db,type=type,spectra=spectra)
                    so_spectra.write_ps('%s/sim_spectra_unbin_%s_%04d.dat'%(simSpectraDir,spec_name,iii),l,cl,type=type,spectra=spectra)

    print ('sim %04d take %.02f seconds to compute'%(iii,time.time()-t0))
