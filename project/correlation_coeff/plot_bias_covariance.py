"""
    This script is used for producing the figure of the paper displaying the comparison of the bias and covariance of the correlation coefficient with the analytical expectation
    To run it: python plot_bias_covariance.py global.dict
"""

import numpy as np
import pylab as plt
from pspy import so_dict,so_spectra,pspy_utils,so_map
import os,sys
import planck_utils
import matplotlib as mpl
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

figure_dir='figures'
theoryFgDir='theory_and_fg'

pspy_utils.create_directory(figure_dir)

iStart=d['iStart']
iStop=d['iStop']
binning_file=d['binning_file']
include_sys=d['include_systematics']
include_foregrounds=d['include_foregrounds']
freqs=d['freqs']
lthmax=1600



if d['use_ffp10']==True:
    mc_dir='monteCarlo_ffp10'
    plot_name='bias_and_cov_ffp10'
else:
    mc_dir='monteCarlo'
    plot_name='bias_and_cov'


freq_pairs=[]
for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        freq_pairs+=[[freq1,freq2]]


lth,psth= pspy_utils.ps_lensed_theory_to_dict(d['theoryfile'],output_type='Cl',lmax=lthmax,lstart=2)


plt.figure(figsize=(15,14))
color_array=['red','gray','orange','blue','green','purple']

count=0
for fpair,color in zip(freq_pairs,color_array):
    
    f0,f1=fpair
    fname='%sx%s'%(f0,f1)
    cl={}
    error={}
    mc_error={}
    model={}
    lmin,lmax=d['lrange_%sx%s'%(f0,f1)]

    for spec in ['TT','EE','TE']:
        if include_foregrounds==True:
            lth,model[spec,fname]=np.loadtxt('%s/clth_fg_%s_%s.dat'%(theoryFgDir,fname,spec),unpack=True)
        else:
            model[spec,fname]=psth[spec]
        
        lb,model[spec,fname]= planck_utils.binning(lth, model[spec,fname],lthmax,binning_file=binning_file)
        id=np.where((lb>lmin) &(lb<lmax))
        model[spec,fname]=model[spec,fname][id]

    model['r',fname]=model['TE',fname]/np.sqrt(model['TT',fname]*model['EE',fname])

    cov_TTTT=np.loadtxt('%s/diagonal_select_cov_mat_TT_%s_%s_TT_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_EEEE=np.loadtxt('%s/diagonal_select_cov_mat_EE_%s_%s_EE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_TETE=np.loadtxt('%s/diagonal_select_cov_mat_TE_%s_%s_TE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))

    cov_TTEE=np.loadtxt('%s/diagonal_select_cov_mat_TT_%s_%s_EE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_TTTE=np.loadtxt('%s/diagonal_select_cov_mat_TT_%s_%s_TE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))
    cov_EETE=np.loadtxt('%s/diagonal_select_cov_mat_EE_%s_%s_TE_%s_%s.dat'%(mc_dir,fname,'hm1xhm2',fname,'hm1xhm2'))



    l,r,std_r=np.loadtxt('%s/spectra_r_%s_hm1xhm2.dat'%(mc_dir,fname),unpack=True)

    mc_std_r=std_r/np.sqrt(iStop-iStart)
        
    bias_r=r-model['r',fname]

    bias_r_th=3./8*(cov_EEEE/(model['EE',fname])**2+ cov_TTTT/(model['TT',fname])**2)
    bias_r_th+=1./4*cov_TTEE/(model['EE',fname]*model['TT',fname])
    bias_r_th-=1./2*(cov_TTTE/(model['TE',fname]*model['TT',fname])+ cov_EETE/(model['TE',fname]*model['EE',fname]))
    bias_r_th*=model['r',fname]

    std_r_th=cov_TETE/(model['TE',fname])**2+1./4*cov_TTTT/(model['TT',fname])**2+1./4*cov_EEEE/(model['EE',fname])**2
    std_r_th-=(cov_TTTE/(model['TE',fname]*model['TT',fname])+ cov_EETE/(model['TE',fname]*model['EE',fname]))
    std_r_th+=1/2*cov_TTEE/(model['EE',fname]*model['TT',fname])
    std_r_th=np.sqrt(std_r_th*model['r',fname]**2)

    print(count,2*count)

    plt.subplot(6,2,1+2*count)

    plt.errorbar(l,std_r,fmt='.',color=color)
    plt.plot(l,std_r_th,color=color,label='%s GHz x %s GHz'%(f0,f1))
    if count==0:
        plt.title(r'$\sigma ({\cal R}^{TE}_{\ell})$',fontsize=22)
    if count==5:
        plt.xlabel(r'$\ell$',fontsize=22)
    plt.legend(fontsize=12,loc='upper center',frameon=False)


    plt.subplot(6,2,2*(count+1))

    plt.errorbar(l,bias_r,mc_std_r,fmt='.',color=color)
    plt.plot(l,bias_r_th,color=color)
    if count==0:
        plt.title(r'$\alpha_{\ell}{\cal R}^{TE}_{\ell}$',fontsize=22)
    if count==5:
        plt.xlabel(r'$\ell$',fontsize=22)
    count+=1

#plt.show()
plt.savefig('%s/%s.pdf'%(figure_dir,plot_name),bbox_inches = 'tight')
plt.clf()
plt.close()








