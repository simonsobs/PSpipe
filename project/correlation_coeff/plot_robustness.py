"""
    This script is used for producing the figure of the paper displaying the robustness of the correlation coefficient with respect to systematics
    To run it: python plot_robustness.py global_syst.dict
"""

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
    simSpectraDir='sim_spectra_syst'
    mc_dir='monteCarlo_syst'
else:
    simSpectraDir='sim_spectra'
    mc_dir='monteCarlo'

freq_pairs=[]
for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        freq_pairs+=[[freq1,freq2]]

lth,psth= pspy_utils.ps_lensed_theory_to_dict(d['theoryfile'],output_type='Cl',lmax=lthmax,lstart=2)

for fpair in freq_pairs:
    
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

    ylim={}
    ylim[1]=[-0.015,0.01]
    ylim[2]=[-0.0005,0.0005]
    ylim[3]=[-0.00001,0.00003]
    ylim[4]=[-0.01,0.01]

    ylabel_top={}
    ylabel_bottom={}
    ylabel_top[1]=r'$ \langle \hat{D}^{\rm TT, syst}_{b} \rangle $'
    ylabel_top[2]=r'$ \langle \hat{D}^{\rm TE, syst}_{b} \rangle$'
    ylabel_top[3]=r'$ \langle \hat{D}^{\rm EE, syst}_{b} \rangle$'
    ylabel_top[4]=r'$ \langle \hat{R}^{\rm TE, syst}_{b} \rangle$'

    ylabel_bottom[1]=r'$ \langle \hat{D}^{\rm TT, syst}_{b} \rangle- D^{\rm TT, th}_{b} $'
    ylabel_bottom[2]=r'$ \langle \hat{D}^{\rm TE, syst}_{b} \rangle- D^{\rm TE, th}_{b}  $'
    ylabel_bottom[3]=r'$ \langle \hat{D}^{\rm EE, syst}_{b} \rangle- D^{\rm EE, th}_{b}  $'
    ylabel_bottom[4]=r'$ \langle \hat{R}^{\rm TE, syst}_{b} \rangle-  R^{\rm TE, th}_{b} $'

    plt.figure(figsize=(20,10))
    count=1

    for spec in ['TT','TE','EE','r']:
    
        l,cl[spec],error[spec]=np.loadtxt('%s/spectra_%s_%s_hm1xhm2.dat'%(mc_dir,spec,fname),unpack=True)
        mc_error[spec]=error[spec]/np.sqrt(iStop-iStart)
        
        chi2=np.sum((cl[spec]-model[spec,fname])**2/mc_error[spec]**2)
        dof=len(l)
        
        if spec=='r':
            fac=1
        else:
            fac=l*(l+1)/(2*np.pi)

        plt.subplot(2,4,count)
        plt.yticks([])
        plt.ylabel(ylabel_top[count],fontsize=20)
        plt.errorbar(l,cl[spec]*fac,mc_error[spec]*fac,fmt='.',color='black',label='recovered mean')
        plt.plot(l,model[spec,fname]*fac,color='grey',label='input theory')
        if count==1:
            plt.legend(fontsize=14)
        plt.subplot(2,4,4+count)
        plt.yticks([])
        plt.ylabel(ylabel_bottom[count],fontsize=20)
        plt.ylim(ylim[count][0], ylim[count][1])
        plt.errorbar(l,l*0,color='grey')
        plt.errorbar(l, (cl[spec]-model[spec,fname]),mc_error[spec],fmt='.',label=r'$\chi^{2}$/DoF= %.0f/%d'%(chi2,dof),color='black')
        plt.xlabel(r'$\ell$',fontsize=18)
        plt.legend(fontsize=14)
        count+=1

    plt.savefig('%s/robustness_%s.pdf'%(figure_dir,fname),bbox_inches = 'tight')
    plt.clf()
    plt.close()










