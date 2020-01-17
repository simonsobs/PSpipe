"""
This script is used to plot the power spectra of the SO simulations.
The code will run as follow (example):
python sim_spectra_plot.py global_sims_all.dict
"""

import matplotlib
matplotlib.use('Agg')
from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
from matplotlib.pyplot import cm
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import h5py

def mat2dict(filename):
    # function to read Alex file format
    mat=np.load(filename)
    return mat[0,0], mat[1,1], mat[2,2], mat[0,1]


# We start by reading the info in the dictionnary

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type=d['type']
experiment=d['experiment']
lmax=d['lmax']
type=d['type']
hdf5=d['hdf5']
clfile=d['clfile']
run_name=d['run_name']
specDir='spectra'

# create a dictionnary to put the plot in

plot_dir='spectra_plot'
pspy_utils.create_directory(plot_dir)

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']


# Read the theory spectra provided by Alex
Dlth={}
Dlth['TT'], Dlth['EE'],Dlth['BB'],Dlth['TE'] =mat2dict(d['theory_file'])
Dlth['ET']=Dlth['TE'].copy()
Dlth['EB']=Dlth['TE']*0
Dlth['TB']=Dlth['TE']*0
Dlth['BE']=Dlth['TE']*0
Dlth['BT']=Dlth['TE']*0

lth=np.arange(2,len(Dlth['TT'])+2)
fth=lth*(lth+1)/(2*np.pi)

for spec in spectra:
    Dlth[spec]*=fth
    Dlth[spec]=Dlth[spec][:lmax]
lth=lth[:lmax]


# Read all spectra and put them in a dictionnary

Db_dict={}
n_spec={}
for kind in ['noise','cross']:
    for spec in spectra:
        n_spec[kind]=0
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
                            
                        spec_name='%s_%s_%s_%sx%s_%s_%s'%(type,run_name,exp1,f1,exp2,f2,kind)

                        if hdf5:
                            lb,Db=so_spectra.read_ps_hdf5(spectra_hdf5,spec_name,spectra=spectra)
                        else:
                            lb,Db=so_spectra.read_ps(specDir+'/%s.dat'%spec_name,spectra=spectra)

                        Db_dict[kind,spec,exp1,f1,exp2,f2]=Db[spec]
                        n_spec[kind]+=1

# Some lcut, the low frequency beam is not very well defined at high ell

lcut={}
lmax_spec={}

lmax_spec['la','027']=3000
lmax_spec['la','039']=3000
lmax_spec['la','093']=lmax
lmax_spec['la','145']=lmax
lmax_spec['la','225']=lmax
lmax_spec['la','280']=lmax

for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue
                
                lcut[exp1,f1,exp2,f2]=np.int((lmax_spec[exp1,f1]+ lmax_spec[exp2,f2])/2)

# we will use both log and linear scale for the plots


plot_type={}
plot_type['TT']=['log','linear']
plot_type['EE']=['log','linear']
plot_type['TE']=['linear']
plot_type['BB']=['log','linear']
plot_type['ET']=['linear']
plot_type['TB']=['linear']
plot_type['BT']=['linear']
plot_type['EB']=['linear']
plot_type['BE']=['linear']

# make the plot

for kind in ['noise','cross']:
    for spec in spectra:
        for fig in plot_type[spec]:
    
            plt.figure(figsize=(12,12))
            color=iter(cm.rainbow(np.linspace(0,1,n_spec['cross']+1)))
    
            if fig=='log':
                plt.semilogy()
    
            exp_name=''

            for id_exp1,exp1 in enumerate(experiment):
                freqs1=d['freq_%s'%exp1]
                for id_f1,f1 in enumerate(freqs1):
                    for id_exp2,exp2 in enumerate(experiment):
                        freqs2=d['freq_%s'%exp2]
                        for id_f2,f2 in enumerate(freqs2):
                            if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                            if  (id_exp1>id_exp2) : continue
                        
                            if f1==f2:
                                marker='.'
                            else:
                                marker='--'
                        
                            lmax= lcut[exp1,f1,exp2,f2]
            
                            mylb=lb.copy()
                            id=np.where(lb<lmax)
                
                            mylb=lb[id]
                            Db_dict[kind,spec,exp1,f1,exp2,f2]=Db_dict[kind,spec,exp1,f1,exp2,f2][id]
                    
                            c=next(color)
                            if (fig=='linear') and (spec=='TT'):
                                plt.errorbar(mylb,Db_dict[kind,spec,exp1,f1,exp2,f2]*mylb**2,fmt=marker,color=c,label='%s %s%s x %s%s'%(kind,exp1,f1,exp2,f2))
                            else:
                                plt.errorbar(mylb,Db_dict[kind,spec,exp1,f1,exp2,f2],fmt=marker,color=c,label='%s %s%s x %s%s'%(kind,exp1,f1,exp2,f2))

                exp_name+='_%s'%exp1
            
            if (fig=='linear') and (spec=='TT'):
                plt.plot(lth[2:lmax],Dlth[spec][2:lmax]*lth[2:lmax]**2,color='grey',alpha=0.4,label='CMB only')
            else:
                plt.plot(lth[2:lmax],Dlth[spec][2:lmax],color='grey',alpha=0.4,label='CMB only')

            if (fig=='log') and (spec=='TT'):
                plt.ylim(10**-4,10**4)
            if (fig=='linear') and (spec=='TT'):
                plt.ylim(-1*10**9,4*10**9)
                
            if (fig=='log') and (spec=='EE'):
                plt.ylim(10**-4,60)
            if (fig=='linear') and (spec=='EE'):
                plt.ylim(10**-4,60)
                            
            if (fig=='log') and (spec=='TE'):
                plt.ylim(10**-3,150)
            if (fig=='linear') and (spec=='TE'):
                plt.ylim(-150,150)
            
            if (fig=='log') and (spec=='BB'):
                plt.ylim(10**-4,40)
            if (fig=='linear') and (spec=='BB'):
                plt.ylim(10**-4,5)

            if fig=='log':
                plt.legend(fontsize=14,bbox_to_anchor=(1.4, 1.1))
            else:
                plt.legend(fontsize=14,bbox_to_anchor=(1.4, 1.))

            plt.xlabel(r'$\ell$',fontsize=20)
            if (fig=='linear') and (spec=='TT'):
                plt.ylabel(r'$\ell^{2} D^{TT}_\ell$',fontsize=20)
            else:
                plt.ylabel(r'$ D^{%s}_\ell$'%spec,fontsize=20)

            plt.savefig('%s/all_%s_%s_%s_spectra_%s_all%s.png'%(plot_dir,run_name,fig,kind,spec,exp_name),bbox_inches='tight')
            plt.clf()
            plt.close()


