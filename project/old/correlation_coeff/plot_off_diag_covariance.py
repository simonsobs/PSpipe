

import numpy as np
import pylab as plt
from pspy import so_dict,so_spectra,pspy_utils,so_map
import os,sys
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

figure_dir='figures'
pspy_utils.create_directory(figure_dir)

iStart=d['iStart']
iStop=d['iStop']
binning_file=d['binning_file']
include_sys=d['include_systematics']
freqs=d['freqs']
lthmax=1600

if include_sys==True:
    mc_dir='monteCarlo_syst'
    plot_name='robustness'
else:
    mc_dir='monteCarlo'
    plot_name='bias'

freq_pairs=[]
for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        freq_pairs+=[[freq1,freq2]]


lth,psth= pspy_utils.ps_lensed_theory_to_dict(d['theoryfile'],output_type='Cl',lmax=lthmax,lstart=2)


plt.figure(figsize=(18,10))
color_array=['red','blue','green','gray','purple','orange']

for fpair,color in zip(freq_pairs,color_array):
    
    f0,f1=fpair
    fname='%sx%s'%(f0,f1)
    cl={}
    error={}
    mc_error={}
    model={}
    lmin,lmax=d['lrange_%sx%s'%(f0,f1)]

    for spec in ['TT','EE','TE']:
        model[spec,fname]=psth[spec]
        lb,model[spec,fname]= planck_utils.binning(lth, model[spec,fname],lthmax,binning_file=binning_file)
        id=np.where((lb>lmin) &(lb<lmax))
        model[spec,fname]=model[spec,fname][id]

    model['r',fname]=model['TE',fname]/np.sqrt(model['TT',fname]*model['EE',fname])

    l,r,std_r=np.loadtxt('%s/spectra_r_%s_hm1xhm2.dat'%(mc_dir,fname),unpack=True)


    cov_TTTT=np.loadtxt('%s/diagonal_select_cov_mat_TT_%s_%s_TT_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_EEEE=np.loadtxt('%s/diagonal_select_cov_mat_EE_%s_%s_EE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_TETE=np.loadtxt('%s/diagonal_select_cov_mat_TE_%s_%s_TE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))

    cov_TTEE=np.loadtxt('%s/diagonal_select_cov_mat_TT_%s_%s_EE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_TTTE=np.loadtxt('%s/diagonal_select_cov_mat_TT_%s_%s_TE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_EETE=np.loadtxt('%s/diagonal_select_cov_mat_EE_%s_%s_TE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))

    cov_TTr=np.loadtxt('%s/diagonal_select_cov_mat_TT_%s_%s_r_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_EEr=np.loadtxt('%s/diagonal_select_cov_mat_EE_%s_%s_r_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))

    cov_rr=np.loadtxt('%s/diagonal_select_cov_mat_r_%s_%s_r_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))


    cov_TTr_th=model['r',fname]*(cov_TTTE/model['TE',fname]-1/2*(cov_TTTT/model['TT',fname]+cov_TTEE/model['EE',fname]))
    cov_EEr_th=model['r',fname]*(cov_EETE/model['TE',fname]-1/2*(cov_EEEE/model['EE',fname]+cov_TTEE/model['TT',fname]))

    plt.plot(l,cov_TTr,'o')
    plt.plot(l,cov_TTr_th)
    plt.show()

    plt.plot(l,cov_EEr,'o')
    plt.plot(l,cov_EEr_th)
    plt.show()


