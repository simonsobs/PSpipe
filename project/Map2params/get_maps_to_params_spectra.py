from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
import  numpy as np, pylab as plt, healpy as hp
import os,sys
import so_noise_calculator_public_20180822 as noise_calc
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

freqs=d['freqs']
lmax=d['lmax']
niter=d['niter']
nSplits=d['nSplits']
type=d['type']
binning_file=d['binning_file']
lcut=d['lcut']

window_dir='window'
mcm_dir='mcm'
noise_dir='noise_ps'
specDir='spectra'
lmax_simu=lmax

spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'w')


if d['spin']=='0-2':
    ncomp=3
    spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']
    l,Nl_array_T,Nl_array_P=maps_to_params_utils.get_noise_matrix_spin0and2(noise_dir,freqs,lmax_simu+1,nSplits,lcut=lcut)

elif d['spin']=='0':
    ncomp=1
    spectra=None
    spin_pairs=None
    l,Nl_array_T=maps_to_params_utils.get_noise_matrix_spin0(noise_dir,freqs,lmax_simu+1,nSplits,lcut=lcut)
    Nl_array_P=None


template=so_map.car_template(ncomp,d['ra0'],d['ra1'],d['dec0'],d['dec1'],d['res'])
ps=powspec.read_spectrum(d['clfile'])[:ncomp,:ncomp]

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d['iStart'], imax=d['iStop'])


time=time.time()
for iii in subtasks:
    
    
    alms= curvedsky.rand_alm(ps, lmax=lmax_simu)
    nlms=maps_to_params_utils.generate_noise_alms(Nl_array_T,Nl_array_P,lmax_simu,nSplits,ncomp)
    master_alms={}
    
    for fid,f in enumerate(freqs):
        window=so_map.read_map('%s/window_%s.fits'%(window_dir,f))
        window_tuple=(window,window)
        l,bl= np.loadtxt('beam/beam_%s.dat'%f,unpack=True)
        alms_convolved=maps_to_params_utils.convolved_alms(alms,bl,ncomp)
        for k in range(nSplits):
            noisy_alms=alms_convolved.copy()
            if ncomp==1:
                noisy_alms+=nlms[k][fid]
            elif ncomp==3:
                noisy_alms[0] +=  nlms['T',k][fid]
                noisy_alms[1] +=  nlms['E',k][fid]
                noisy_alms[2] +=  nlms['B',k][fid]

            split=sph_tools.alm2map(noisy_alms,template)
            split=maps_to_params_utils.remove_mean(split,window_tuple,ncomp)
            master_alms[f,k]= sph_tools.get_alms(split,window_tuple,niter,lmax)

    Db_dict={}
    for fid1,f1 in enumerate(freqs):
        for fid2,f2 in enumerate(freqs):
            if fid1>fid2: continue
            for spec in spectra:
                Db_dict[f1,f2,spec,'auto']=[]
                Db_dict[f1,f2,spec,'cross']=[]

            prefix= '%s/%sx%s'%(mcm_dir,f1,f2)
            for s1 in range(nSplits):
                for s2 in range(nSplits):
                    if (s1>s2) & (fid1==fid2): continue
                    mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

                    l,ps_master= so_spectra.get_spectra(master_alms[f1,s1],master_alms[f2,s2],spectra=spectra)
                    spec_name='%s_%sx%s_%sx%s_%05d'%(type,f1,f2,s1,s2,iii)
                    lb,Db=so_spectra.bin_spectra(l,ps_master,binning_file,lmax,type=type,mbb_inv=mbb_inv,spectra=spectra)
                    so_spectra.write_ps_hdf5(spectra_hdf5,spec_name,lb,Db,spectra=spectra)
                    for spec in spectra:
                        if (s1==s2):
                            Db_dict[f1,f2,spec,'auto']+=[Db[spec]]
                        else:
                            Db_dict[f1,f2,spec,'cross']+=[Db[spec]]

            Db_dict_auto={}
            Db_dict_cross={}
            nb={}

            for spec in spectra:
                Db_dict_auto[spec]=np.mean(Db_dict[f1,f2,spec,'auto'],axis=0)
                spec_name_auto='%s_%sx%s_auto_%05d'%(type,f1,f2,iii)
                Db_dict_cross[spec]=np.mean(Db_dict[f1,f2,spec,'cross'],axis=0)
                spec_name_cross='%s_%sx%s_cross_%05d'%(type,f1,f2,iii)
                nb[spec]= (Db_dict_auto[spec]- Db_dict_cross[spec])/d['nSplits']
                spec_name_noise='%s_%sx%s_noise_%05d'%(type,f1,f2,iii)
        
            so_spectra.write_ps_hdf5(spectra_hdf5,spec_name_auto,lb,Db_dict_auto,spectra=spectra)
            so_spectra.write_ps_hdf5(spectra_hdf5,spec_name_cross,lb,Db_dict_cross,spectra=spectra)
            so_spectra.write_ps_hdf5(spectra_hdf5,spec_name_noise,lb,nb,spectra=spectra)
    print ('sim number %05d done in .02f s'%(iii,time.time()-t))

