#import matplotlib
#matplotlib.use('Agg')
import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap
import time
import planck_utils



d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

auxMapDir='window'
mcmDir='mcm'
spectraDir='spectra'

try:
    os.makedirs(spectraDir)
except:
    pass

spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

arrays=d['arrays']
niter=d['niter']
lmax=d['lmax']
type=d['type']
binning_file=d['binning_file']
remove_mono_dipo_T=d['remove_mono_dipo_T']
remove_mono_dipo_pol=d['remove_mono_dipo_pol']


experiment='Planck'

alms={}
nsplit={}

split=['hm1','hm2']

for ar in arrays:
    maps=d['map_%s'%ar]
    for hm,map in zip(split,maps):
        
        window_T=so_map.read_map('%s/window_T_%s_%s-%s.fits'%(auxMapDir,experiment,ar,hm))
        window_pol=so_map.read_map('%s/window_P_%s_%s-%s.fits'%(auxMapDir,experiment,ar,hm))
        window_tuple=(window_T,window_pol)
        del window_T,window_pol

        pl_map=so_map.read_map('%s'%map,fields_healpix=(0,1,2))
        pl_map.data*=10**6
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
                so_spectra.write_ps('%s/spectra_%s.dat'%(spectraDir,spec_name),lb,Db,type=type,spectra=spectra)
                so_spectra.write_ps('%s/spectra_unbin_%s.dat'%(spectraDir,spec_name),l,cl,type=type,spectra=spectra)


