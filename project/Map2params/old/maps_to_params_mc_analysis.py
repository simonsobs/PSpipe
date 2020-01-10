from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
from matplotlib.pyplot import cm
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type=d['type']
experiment=d['experiment']
iStart=d['iStart']
iStop=d['iStop']
lmax=d['lmax']
type=d['type']
clfile=d['clfile']
lcut=d['lcut']
hdf5=d['hdf5']
multistep_path=d['multistep_path']

foreground_dir=d['foreground_dir']
extragal_foregrounds=d['extragal_foregrounds']

specDir='spectra'

if hdf5:
    spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'r')

mcm_dir='mcm'
mc_dir='monteCarlo'

pspy_utils.create_directory(mc_dir)

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']


for kind in ['cross','noise','auto']:
    vec_list=[]
    vec_list_restricted=[]

    for iii in range(iStart,iStop):
        vec=[]
        vec_restricted=[]
        count=0
        for spec in spectra:
            for id_exp1,exp1 in enumerate(experiment):
                freqs1=d['freq_%s'%exp1]
                for id_f1,f1 in enumerate(freqs1):
                    for id_exp2,exp2 in enumerate(experiment):
                        freqs2=d['freq_%s'%exp2]
                        for id_f2,f2 in enumerate(freqs2):
                            if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                            if  (id_exp1>id_exp2) : continue
                            if (exp1!=exp2) & (kind=='noise'): continue
                            if (exp1!=exp2) & (kind=='auto'): continue

                            spec_name='%s_%s_%sx%s_%s_%s_%05d'%(type,exp1,f1,exp2,f2,kind,iii)
                    
                            if hdf5:
                                lb,Db=so_spectra.read_ps_hdf5(spectra_hdf5,spec_name,spectra=spectra)
                            else:
                                lb,Db=so_spectra.read_ps(specDir+'/%s.dat'%spec_name,spectra=spectra)

                            n_bins=len(lb)
                            vec=np.append(vec,Db[spec])
                            if spec=='TT' or spec=='EE':
                                vec_restricted=np.append(vec_restricted,Db[spec])
                            if spec=='TE':
                                vec_restricted=np.append(vec_restricted,(Db['TE']+Db['ET'])/2)

                                
        vec_list+=[vec]
        vec_list_restricted+=[vec_restricted]

    mean_vec=np.mean(vec_list,axis=0)
    mean_vec_restricted=np.mean(vec_list_restricted,axis=0)

    cov=0
    cov_restricted=0

    for iii in range(iStart,iStop):
        cov+=np.outer(vec_list[iii],vec_list[iii])
        cov_restricted+=np.outer(vec_list_restricted[iii],vec_list_restricted[iii])

    cov=cov/(iStop-iStart)-np.outer(mean_vec, mean_vec)
    cov_restricted=cov_restricted/(iStop-iStart)-np.outer(mean_vec_restricted, mean_vec_restricted)


    np.save('%s/cov_all_%s.npy'%(mc_dir,kind),cov)
    np.save('%s/cov_restricted_all_%s.npy'%(mc_dir,kind),cov_restricted)

    id_spec=0
    for spec in spectra:
        for id_exp1,exp1 in enumerate(experiment):
            freqs1=d['freq_%s'%exp1]
            for id_f1,f1 in enumerate(freqs1):
                for id_exp2,exp2 in enumerate(experiment):
                    freqs2=d['freq_%s'%exp2]
                    for id_f2,f2 in enumerate(freqs2):
                        if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                        if  (id_exp1>id_exp2) : continue
                        if (exp1!=exp2) & (kind=='noise'): continue
                        if (exp1!=exp2) & (kind=='auto'): continue
                
                        mean=mean_vec[id_spec*n_bins:(id_spec+1)*n_bins]
                        std=np.sqrt(cov[id_spec*n_bins:(id_spec+1)*n_bins,id_spec*n_bins:(id_spec+1)*n_bins].diagonal())
                        
                        np.savetxt('%s/spectra_%s_%s_%sx%s_%s_%s.dat'%(mc_dir,spec,exp1,f1,exp2,f2,kind), np.array([lb,mean,std]).T)
                        id_spec+=1

