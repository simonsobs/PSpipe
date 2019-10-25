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

experiment=d['experiment']
lmax=d['lmax']
niter=d['niter']
type=d['type']
binning_file=d['binning_file']
lcut=d['lcut']
hdf5=d['hdf5']
writeAll=d['writeAll']

include_fg=d['include_fg']
fg_dir=d['fg_dir']
fg_components=d['fg_components']

window_dir='window'
mcm_dir='mcm'
noise_dir='noise_ps'
specDir='spectra'

lmax_simu=lmax

if hdf5:
    spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'w')
else:
    pspy_utils.create_directory(specDir)

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

allfreqs=[]
for exp in experiment:
    freqs=d['freq_%s'%exp]
    for freq in freqs:
        allfreqs+=[freq]

ps=powspec.read_spectrum(d['clfile'])[:ncomp,:ncomp]

if include_fg==True:
    l,ps_extragal=maps_to_params_utils.get_foreground_matrix(fg_dir,fg_components,allfreqs,lmax_simu+1)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d['iStart'], imax=d['iStop'])

for iii in subtasks:
    t0=time.time()
    
    alms= curvedsky.rand_alm(ps, lmax=lmax_simu)
    
    if include_fg==True:
        flms=curvedsky.rand_alm(ps_extragal,lmax=lmax_simu)

    master_alms={}
    
    fcount=0
    for exp in experiment:
        
        freqs=d['freq_%s'%exp]
        nSplits=d['nSplits_%s'%exp]
        
        if d['pixel_%s'%exp]=='CAR':
            template=so_map.car_template(ncomp,d['ra0_%s'%exp],d['ra1_%s'%exp],d['dec0_%s'%exp],d['dec1_%s'%exp],d['res_%s'%exp])
        elif d['pixel_%s'%exp]=='HEALPIX':
            template=so_map.healpix_template(ncomp,nside=d['nside_%s'%exp])

        l,Nl_array_T,Nl_array_P=maps_to_params_utils.get_noise_matrix_spin0and2(noise_dir,exp,freqs,lmax_simu+1,nSplits,lcut=lcut)
        
        nlms=maps_to_params_utils.generate_noise_alms(Nl_array_T,Nl_array_P,lmax_simu,nSplits,ncomp)
    
        for fid,f in enumerate(freqs):
            window=so_map.read_map('%s/window_%s_%s.fits'%(window_dir,exp,f))
            window_tuple=(window,window)
            l,bl= np.loadtxt('beam/beam_%s_%s.dat'%(exp,f),unpack=True)
            
            alms_convolved=alms.copy()
            if include_fg==True:
                alms_convolved[0]+=flms[fcount]
            
            alms_convolved=maps_to_params_utils.convolved_alms(alms_convolved,bl,ncomp)
            
            fcount+=1
            
            for k in range(nSplits):
                noisy_alms=alms_convolved.copy()
                
                noisy_alms[0] +=  nlms['T',k][fid]
                noisy_alms[1] +=  nlms['E',k][fid]
                noisy_alms[2] +=  nlms['B',k][fid]

                split=sph_tools.alm2map(noisy_alms,template)
                split=maps_to_params_utils.remove_mean(split,window_tuple,ncomp)
                master_alms[exp,f,k]= sph_tools.get_alms(split,window_tuple,niter,lmax)

    Db_dict={}
    for id_exp1,exp1 in enumerate(experiment):
        freqs1=d['freq_%s'%exp1]
        nSplits1=d['nSplits_%s'%exp1]
        for id_f1,f1 in enumerate(freqs1):
            for id_exp2,exp2 in enumerate(experiment):
                freqs2=d['freq_%s'%exp2]
                nSplits2=d['nSplits_%s'%exp2]

                for id_f2,f2 in enumerate(freqs2):

                    if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                    if  (id_exp1>id_exp2) : continue

    
                    for spec in spectra:
                        Db_dict[exp1,f1,exp2,f2,spec,'auto']=[]
                        Db_dict[exp1,f1,exp2,f2,spec,'cross']=[]
                    
                    
                    prefix= '%s/%s_%sx%s_%s'%(mcm_dir,exp1,f1,exp2,f2)
                    for s1 in range(nSplits1):
                        for s2 in range(nSplits2):

                            mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

                            l,ps_master= so_spectra.get_spectra(master_alms[exp1,f1,s1],master_alms[exp2,f2,s2],spectra=spectra)
                            spec_name='%s_%s_%s_%dx%s_%s_%d_%05d'%(type,exp1,f1,s1,exp2,f2,s2,iii)
                            lb,Db=so_spectra.bin_spectra(l,ps_master,binning_file,lmax,type=type,mbb_inv=mbb_inv,spectra=spectra)
                    
                            if writeAll:
                                if hdf5:
                                    so_spectra.write_ps_hdf5(spectra_hdf5,spec_name,lb,Db,spectra=spectra)
                                else:
                                    so_spectra.write_ps(specDir+'/%s.dat'%spec_name,lb,Db,type,spectra=spectra)

                            for spec in spectra:
                                if (s1==s2) & (exp1==exp2):
                                    if spec=='TT':
                                        print ('auto %s_%s split%d X %s_%s split%d'%(exp1,f1,s1,exp2,f2,s2))
                                    Db_dict[exp1,f1,exp2,f2,spec,'auto']+=[Db[spec]]
                                else:
                                    if spec=='TT':
                                        print ('cross %s_%s split%d X %s_%s split%d'%(exp1,f1,s1,exp2,f2,s2))
                                    
                                    Db_dict[exp1,f1,exp2,f2,spec,'cross']+=[Db[spec]]

                    Db_dict_auto={}
                    Db_dict_cross={}
                    nb={}

                    for spec in spectra:
                        
                        Db_dict_cross[spec]=np.mean(Db_dict[exp1,f1,exp2,f2,spec,'cross'],axis=0)
                        spec_name_cross='%s_%s_%sx%s_%s_cross_%05d'%(type,exp1,f1,exp2,f2,iii)
                        
                        if exp1==exp2:
                            Db_dict_auto[spec]=np.mean(Db_dict[exp1,f1,exp2,f2,spec,'auto'],axis=0)
                            spec_name_auto='%s_%s_%sx%s_%s_auto_%05d'%(type,exp1,f1,exp2,f2,iii)
                            nb[spec]= (Db_dict_auto[spec]- Db_dict_cross[spec])/d['nSplits_%s'%exp]
                            spec_name_noise='%s_%s_%sx%s_%s_noise_%05d'%(type,exp1,f1,exp2,f2,iii)

                    if hdf5:
                        so_spectra.write_ps_hdf5(spectra_hdf5,spec_name_cross,lb,Db_dict_cross,spectra=spectra)
                        if exp1==exp2:
                            so_spectra.write_ps_hdf5(spectra_hdf5,spec_name_auto,lb,Db_dict_auto,spectra=spectra)
                            so_spectra.write_ps_hdf5(spectra_hdf5,spec_name_noise,lb,nb,spectra=spectra)
                    
                    else:
                        so_spectra.write_ps(specDir+'/%s.dat'%spec_name_cross,lb,Db_dict_cross,type,spectra=spectra)
                        if exp1==exp2:
                            so_spectra.write_ps(specDir+'/%s.dat'%spec_name_auto,lb,Db_dict_auto,type,spectra=spectra)
                            so_spectra.write_ps(specDir+'/%s.dat'%spec_name_noise,lb,nb,type,spectra=spectra)

                
    print ('sim number %05d done in %.02f s'%(iii,time.time()-t0))

if hdf5:
    spectra_hdf5.close()
