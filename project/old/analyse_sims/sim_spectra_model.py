"""
This script is used for the modeling the theoretical spectra that goes into the covariance computation.
At the moment, we simply used interpolation and extrapolation of the signal only spectra, we hope to change this to actual model spectra in the future.
python sim_spectra_model.py global_sims_all.dict
"""

from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
from matplotlib.pyplot import cm
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import h5py
import scipy.interpolate

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type=d['type']
experiment=d['experiment']
lmax=d['lmax']
type=d['type']
run_name=d['run_name']

specDir='spectra'


ps_model_dir='model'
pspy_utils.create_directory(ps_model_dir)


spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

foreground={}
for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue
                    
                spec_name_combined='%s_combined_%s_%sx%s_%s_cross'%(type,exp1,f1,exp2,f2)
                spec_name_noise='%s_combined_noisy_%s_%sx%s_%s_noise'%(type,exp1,f1,exp2,f2)

                lb,Db_combined=so_spectra.read_ps(specDir+'/%s.dat'%spec_name_combined,spectra=spectra)
                lb,Nb=so_spectra.read_ps(specDir+'/%s.dat'%spec_name_noise,spectra=spectra)

                Dl_interpolate={}
                Nl_interpolate={}
                
                l,bl1= np.loadtxt(d['beam_%s_%s'%(exp1,f1)],unpack=True)
                l,bl2= np.loadtxt(d['beam_%s_%s'%(exp2,f2)],unpack=True)
                bl1,bl2= bl1[:lmax],bl2[:lmax]

                for spec in spectra:
                        
                    l=np.arange(lmax)
                        
                    if spec=='TT' or spec=='TE' or spec=='ET' or spec=='EE' or spec=='BB':
                        Dl = scipy.interpolate.interp1d(lb,Db_combined[spec], fill_value='extrapolate')
                        Dl_interpolate[spec]=np.array([Dl(i) for i in l])
                        Dl_interpolate[spec][:2]=0
                    else:
                        Dl_interpolate[spec]=np.zeros(len(l))


                    if (spec=='TT' or spec=='EE' or spec=='BB') & (f1==f2) & (exp1==exp2):

                        Nl = scipy.interpolate.interp1d(lb,Nb[spec], fill_value='extrapolate')
                        Nl_interpolate[spec]=np.array([Nl(i) for i in l])
                        Nl_interpolate[spec][:30]=0
                        Nl_interpolate[spec]*=bl1*bl2

                    else:
                        Nl_interpolate[spec]=np.zeros(len(l))

                spec_name_combined='%s_%s_%sx%s_%s_cross'%(type,exp1,f1,exp2,f2)
                so_spectra.write_ps(ps_model_dir+'/%s.dat'%spec_name_combined,l,Dl_interpolate,type,spectra=spectra)
                spec_name_noise='%s_%s_%sx%s_%s_noise'%(type,exp1,f1,exp2,f2)
                so_spectra.write_ps(ps_model_dir+'/%s.dat'%spec_name_noise,l,Nl_interpolate,type,spectra=spectra)



