'''
This script is used to get estimate of the planck noise power spectra
It take the unbinned power spectra, bin them with large bin and interpolate between the bin
The logic here is to reduce the scatter in the measured noise power spectra
'''

import matplotlib
matplotlib.use('Agg')
import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap
import time
import scipy.interpolate
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

auxMapDir='window'
mcmDir='mcm'
spectraDir='spectra'

ps_model_dir='model'
plot_dir='plot'

pspy_utils.create_directory(ps_model_dir)
pspy_utils.create_directory(plot_dir)

spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
type=d['type']
freqs=d['freqs']
binning_file=d['binning_file']
lmax=d['lmax']
splits=['hm1','hm2']
size=d['noise_binning_size']
experiment='Planck'

lth = np.arange(2, lmax+2)

bl={}
for freq in freqs:
    for hm in splits:
        lbeam,bl_T= np.loadtxt(d['beam_%s_%s_T'%(freq,hm)],unpack=True)
        bl[freq,hm,'TT']=bl_T[:lmax-2]
        lbeam,bl_pol= np.loadtxt(d['beam_%s_%s_pol'%(freq,hm)],unpack=True)
        bl[freq,hm,'EE']=bl_pol[:lmax-2]
        bl[freq,hm,'BB']= bl[freq,hm,'EE']


ps_dict={}
nl_hm1={}
nl_hm2={}
nl_mean={}
nlth={}

for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        
        for s1,hm1 in enumerate(splits):
            for s2,hm2 in enumerate(splits):
                if (s1>s2) & (c1==c2): continue
                
                spec_name='%s_%sx%s_%s-%sx%s'%(experiment,freq1,experiment,freq2,hm1,hm2)

                l,ps= so_spectra.read_ps('%s/spectra_unbin_%s.dat'%(spectraDir,spec_name),spectra=spectra)
                for spec in spectra:
                    ps_dict[freq1,freq2,spec,hm1,hm2]=ps[spec]

        for spec in spectra:
            
            if (spec=='TT' or spec=='EE' or spec=='BB') & (freq1==freq2):
                
                if spec=='TT':
                    sigma_th=d['sigma_th_%s'%freq1]
                if spec=='EE' or spec=='BB':
                    sigma_th=d['sigma_pol_th_%s'%freq1]
                sigma_th = np.deg2rad(sigma_th)/60
                
                nlth[spec]=lth*0+2*sigma_th**2
                
                bl_hm1=bl[freq1,'hm1',spec]
                bl_hm2=bl[freq1,'hm2',spec]

                lb,nb_hm1= planck_utils.binning(l,ps_dict[freq1,freq2,spec,'hm1','hm1']*bl_hm1**2-ps_dict[freq1,freq2,spec,'hm1','hm2']*bl_hm1*bl_hm2,lmax,size=size)
                lb,nb_hm2= planck_utils.binning(l,ps_dict[freq1,freq2,spec,'hm2','hm2']*bl_hm2**2-ps_dict[freq1,freq2,spec,'hm1','hm2']*bl_hm1*bl_hm2,lmax,size=size)

                f=lb**2/(2*np.pi)

                nl_interpol_1 = scipy.interpolate.interp1d(lb,nb_hm1, fill_value='extrapolate')
                nl_interpol_2 = scipy.interpolate.interp1d(lb,nb_hm2, fill_value='extrapolate')
                nl_interpol_mean= scipy.interpolate.interp1d(lb,(nb_hm1+nb_hm2)/2, fill_value='extrapolate')
                
                nl_hm1[spec]=np.array([nl_interpol_1(i) for i in lth])
                nl_hm2[spec]=np.array([nl_interpol_2(i) for i in lth])
                nl_mean[spec]=np.array([nl_interpol_mean(i) for i in lth])

                if spec=='TT':
                    plt.ylim(-0.002,0.002)
                plt.plot(lth,nlth[spec],label='white noise %sx%s %s'%(freq1,freq2,spec))
                plt.plot(lth,nl_hm1[spec],color='red')
                plt.plot(lth,nl_hm2[spec],color='blue')
                plt.plot(lth,nl_mean[spec],color='black',label='mean')
                plt.plot(lb,nb_hm1,'.',label='hm1xhm1-hm1xhm2',color='red')
                plt.plot(lb,nb_hm2,'.',label='hm2xhm2-hm1xhm2',color='blue')
                plt.legend()
                plt.savefig('%s/noise_interpolate_%s_%s_%s.png'%(plot_dir,freq1,freq2,spec),bbox_inches='tight')
                plt.clf()
                plt.close()
      
            else:
                nl_hm1[spec]=np.zeros(len(lth))
                nl_hm2[spec]=np.zeros(len(lth))
                nl_mean[spec]=np.zeros(len(lth))


        np.savetxt('%s/noise_T_hm1_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nl_hm1['TT']]))
        np.savetxt('%s/noise_P_hm1_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nl_hm1['EE']]))

        np.savetxt('%s/noise_T_hm2_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nl_hm2['TT']]))
        np.savetxt('%s/noise_P_hm2_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nl_hm2['EE']]))

        np.savetxt('%s/noise_T_mean_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nl_mean['TT']/2]))
        np.savetxt('%s/noise_P_mean_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nl_mean['EE']/2]))
        
        np.savetxt('%s/noise_T_th_mean_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nlth['TT']/2]))
        np.savetxt('%s/noise_P_th_mean_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,freq1,experiment,freq2),  np.transpose([lth,nlth['EE']/2]))


        spec_name_noise_hm1='hm1_%s_%sx%s_%s_noise'%(experiment,freq1,experiment,freq2)
        spec_name_noise_hm2='hm2_%s_%sx%s_%s_noise'%(experiment,freq1,experiment,freq2)
        spec_name_noise_mean='mean_%s_%sx%s_%s_noise'%(experiment,freq1,experiment,freq2)

        so_spectra.write_ps(ps_model_dir+'/%s.dat'%spec_name_noise_hm1,lth,nl_hm1,type,spectra=spectra)
        so_spectra.write_ps(ps_model_dir+'/%s.dat'%spec_name_noise_hm2,lth,nl_hm2,type,spectra=spectra)
        so_spectra.write_ps(ps_model_dir+'/%s.dat'%spec_name_noise_mean,lth,nl_mean,type,spectra=spectra)



