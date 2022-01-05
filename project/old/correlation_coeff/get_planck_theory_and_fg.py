import matplotlib
matplotlib.use('Agg')
import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap,powspec
import time
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

theoryFgDir='theory_and_fg'

pspy_utils.create_directory(theoryFgDir)
freqs=d['freqs']
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']


freq_pairs=[]
for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        freq_pairs+=[[freq1,freq2]]

clth={}
fg={}

lth,cl_TT,cl_EE,cl_BB,cl_TE=np.loadtxt('theory_file/cosmo2017_10K_acc3_lensedCls.dat',unpack=True)
clth['TT']=cl_TT
clth['TE']=cl_TE
clth['EE']=cl_EE
clth['BB']=cl_BB

lth,fg['TT','100x100'],fg['TT','143x143'],fg['TT','143x217'],fg['TT','217x217'],fg['EE','100x100'],fg['EE','100x143'],fg['EE','100x217'],fg['EE','143x143'],fg['EE','143x217'],fg['EE','217x217'],fg['TE','100x100'],fg['TE','100x143'],fg['TE','100x217'],fg['TE','143x143'],fg['TE','143x217'],fg['TE','217x217']=np.loadtxt('theory_file/base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.plik_foregrounds',unpack=True)

# We don't have a foreground model for the TT 100x143 and 100x217 spectra, we set foreground to zero for these spectra
fg['TT','100x143']=fg['TT','100x100']*0
fg['TT','100x217']=fg['TT','100x100']*0
fg['TE','100x143']=fg['TT','100x100']*0
fg['TE','100x217']=fg['TT','100x100']*0
fg['EE','100x143']=fg['TT','100x100']*0
fg['EE','100x217']=fg['TT','100x100']*0


# The foreground+syst spectra have a very strange shape at high ell, we therefore regularize them
# Note that the scales beyond the regularisation scale will not be used in the paper

lth_max=6000
l_regul=1450
fg_regularised=np.zeros(lth_max)
lth=np.arange(2,lth_max+2)
lth_padded=np.arange(0,lth_max)
fth=lth*(lth+1)/(2*np.pi)
fth_padded=lth_padded*(lth_padded+1)/(2*np.pi)
cl_th_and_fg={}

for spec in ['TT','EE','TE']:
    for f in freq_pairs:
        
        f0,f1=f[0],f[1]
        fname='%sx%s'%(f0,f1)

        fg_regularised[:l_regul]=fg[spec,fname][:l_regul]/fth[:l_regul]
        fg_regularised[l_regul:]=fg_regularised[l_regul-1]
        
        cl_th_and_fg[spec,fname]=np.zeros(lth_max)
        cl_th_and_fg[spec,fname][2:lth_max]=clth[spec][:lth_max-2]/fth[:lth_max-2]+fg_regularised[:lth_max-2]
        np.savetxt('%s/clth_fg_%s_%s.dat'%(theoryFgDir,fname,spec),np.transpose( [lth_padded,cl_th_and_fg[spec,fname]]))
        
        plt.plot(lth_padded,cl_th_and_fg[spec,fname]*fth_padded,label='%s'%fname)
    plt.legend()
    plt.savefig('%s/clth_fg_%s.png'%(theoryFgDir,spec))
    plt.clf()
    plt.close()

for f in freq_pairs:
    f0,f1=f[0],f[1]
    fname='%sx%s'%(f0,f1)
    cl_th_and_fg['ET',fname]=cl_th_and_fg['TE',fname]
    cl_th_and_fg['EB',fname]=cl_th_and_fg['TE',fname]*0
    cl_th_and_fg['BE',fname]=cl_th_and_fg['TE',fname]*0
    cl_th_and_fg['TB',fname]=cl_th_and_fg['TE',fname]*0
    cl_th_and_fg['BT',fname]=cl_th_and_fg['TE',fname]*0
    cl_th_and_fg['BB',fname]=np.zeros(lth_max)
    cl_th_and_fg['BB',fname][2:lth_max]=clth['BB'][:lth_max-2]/fth[:lth_max-2]
    for spec in spectra:
        fname_revert='%sx%s'%(f1,f0)
        cl_th_and_fg[spec,fname_revert]=cl_th_and_fg[spec,fname]

mat=np.zeros((9,9,lth_max))
fields=['T','E','B']
for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        for count1,field1 in enumerate(fields):
            for count2,field2 in enumerate(fields):

                fname='%sx%s'%(freq1,freq2)
                print (count1+3*c1,count2+3*c2, field1+field2,fname)
                mat[count1+3*c1,count2+3*c2,:lth_max]=cl_th_and_fg[field1+field2,fname]


ps_th=powspec.read_spectrum(d['theoryfile'])[:3,:3]


np.save('%s/signal_fg_matrix.npy'%theoryFgDir,mat)
