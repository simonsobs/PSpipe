"""
This script is used to compute all power spectra of the SO simulations.
You can either write the spectra as separate files in a folder or put them all in a single hdf5 file.
The code will run as follow (example):
python sim_spectra.py global_sims_all.dict
"""

from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import h5py
import time


# We start by reading the info in the dictionnary

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiment=d['experiment']
lmax=d['lmax']
niter=d['niter']
type=d['type']
binning_file=d['binning_file']
hdf5=d['hdf5']
writeAll=d['writeAll']
run_name=d['run_name']

def remove_mean(map,window,ncomp):
    # single function to remove the mean of the data before taking the power spectrum
    for i in range(ncomp):
        map.data[0]-=np.mean(map.data[0]*window[0].data)
        map.data[1]-=np.mean(map.data[1]*window[1].data)
        map.data[2]-=np.mean(map.data[2]*window[1].data)
    return map


window_dir='window'
mcm_dir='mcm'
specDir='spectra'

# create spectra folder or initiale hdf5 file


if hdf5:
    spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'w')
else:
    pspy_utils.create_directory(specDir)

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']


# We first compute all the alms of the different split frequency maps (that have been multiplied by their associated window function)
master_alms={}
nSplits={}
for exp in experiment:
    freqs=d['freq_%s'%exp]
    for fid,f in enumerate(freqs):
        
        maps=d['maps_%s_%s'%(exp,f)]
        nSplits[exp]=len(maps)
        window_T=so_map.read_map('%s/window_T_%s_%s.fits'%(window_dir,exp,f))
        window_P=so_map.read_map('%s/window_P_%s_%s.fits'%(window_dir,exp,f))

        window_tuple=(window_T,window_P)
        
        count=0
        for map in maps:
            split=so_map.read_map(map)
            if split.ncomp==1:
                Tsplit=split.copy()
                split= so_map.healpix_template(ncomp=3,nside=split.nside)
                split.data[0]=Tsplit.data
                split.data[1]=Tsplit.data*0
                split.data[2]=Tsplit.data*0
            
            split=remove_mean(split,window_tuple,ncomp)
            master_alms[exp,f,count]= sph_tools.get_alms(split,window_tuple,niter,lmax)
            count+=1

# We then compute the cls from the alms and deconvolve the mcm that take into account the effect of the window function
Db_dict={}
for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    nSplits1=nSplits[exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            nSplits2=nSplits[exp2]
                
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
                        spec_name='%s_%s_%s_%s_%dx%s_%s_%d'%(type,run_name,exp1,f1,s1,exp2,f2,s2)
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
                
                # we combine the different cross spectra and auto spectra together and write them to disk
                # we also write the noise power spectra defined as (auto-cross)/nsplits
                    
                for spec in spectra:
                        
                    Db_dict_cross[spec]=np.mean(Db_dict[exp1,f1,exp2,f2,spec,'cross'],axis=0)
                    spec_name_cross='%s_%s_%s_%sx%s_%s_cross'%(type,run_name,exp1,f1,exp2,f2)
                        
                    if exp1==exp2:
                        Db_dict_auto[spec]=np.mean(Db_dict[exp1,f1,exp2,f2,spec,'auto'],axis=0)
                        spec_name_auto='%s_%s_%s_%sx%s_%s_auto'%(type,run_name,exp1,f1,exp2,f2)
                        nb[spec]= (Db_dict_auto[spec]- Db_dict_cross[spec])/nSplits[exp]
                        spec_name_noise='%s_%s_%s_%sx%s_%s_noise'%(type,run_name,exp1,f1,exp2,f2)
            
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

if hdf5:
    spectra_hdf5.close()
