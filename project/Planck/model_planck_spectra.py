import matplotlib
matplotlib.use('Agg')
import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap
import time
import scipy.interpolate

def get_nlth(lmin, lmax, arrays,spectra,bl):
    nlth={}
    l = np.arange(lmin, lmax)
    for c1,ar1 in enumerate(arrays):
        for c2,ar2 in enumerate(arrays):
            if c1>c2: continue
            for spec in spectra:
                if ar1==ar2:
                    fl=bl[ar1][lmin:lmax]
                    sigma=0
                    if spec=='TT':
                        sigma=d['sigma_th_%s'%ar1]
                    if spec=='EE' or spec=='BB':
                        sigma=d['sigma_pol_th_%s'%ar1]
                    sigma = np.deg2rad(sigma)/60
                    print (spec,sigma,ar1,ar2)
                    nlth[ar1,ar2,spec]=l*0+sigma**2*(hp.pixwin(2048)[:len(l)])**2  #/(fl**2)*l*(l+1)/(2*np.pi)*
                else:
                    nlth[ar1,ar2,spec]=l*0
    return l,nlth

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

auxMapDir='window'
mcmDir='mcm'
spectraDir='spectra'

ps_model_dir='model'
pspy_utils.create_directory(ps_model_dir)


spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
type=d['type']
arrays=d['arrays']
niter=d['niter']
binning_file=d['binning_file']
lmax=d['lmax']
split=['hm1','hm2']
experiment='Planck'


lth,Dlth=pspy_utils.ps_lensed_theory_to_dict(d['theoryfile'],output_type='Dl',lmax=lmax,lstart=2)

bl={}
for ar in arrays:
    beam= np.loadtxt(d['beam_%s'%ar])
    ljunk,bl[ar]=beam[:,0],beam[:,1]


lth,nlth=get_nlth(2, lmax+2, arrays,spectra,bl)

Db_dict={}
nb_dict={}
Nl_interpolate={}
spec_name_list=[]


ylim={}
ylim['TT']=[10**-2,10**6]
ylim['TE']=[-200,200]
ylim['ET']=[-200,200]
ylim['TB']=[-200,200]
ylim['BT']=[-20,20]
ylim['EE']=[10**-2,10**6]
ylim['EB']=[-2,2]
ylim['BE']=[-2,2]
ylim['BB']=[-2,2]
ylim['BB']=[10**-2,10**6]

for c1,ar1 in enumerate(arrays):
    
    beam1= np.loadtxt(d['beam_%s'%ar1])
    l,bl1=beam1[:,0],beam1[:,1]

    for c2,ar2 in enumerate(arrays):
        if c1>c2: continue
        
        beam2= np.loadtxt(d['beam_%s'%ar2])
        l,bl2=beam1[:,0],beam1[:,1]

        
        for spec in spectra:
            Db_dict[ar1,ar2,spec,'auto']=[]
            Db_dict[ar1,ar2,spec,'cross']=[]

        for s1,hm1 in enumerate(split):
            for s2,hm2 in enumerate(split):
                if (s1>s2) & (c1==c2): continue
                
                spec_name='%s_%sx%s_%s-%sx%s'%(experiment,ar1,experiment,ar2,hm1,hm2)

                lb,Db= so_spectra.read_ps('%s/spectra_%s.dat'%(spectraDir,spec_name),spectra=spectra)
                for spec in spectra:
                    if (hm1==hm2):
                        Db_dict[ar1,ar2,spec,'auto']+=[Db[spec]]
                    else:
                        Db_dict[ar1,ar2,spec,'cross']+=[Db[spec]]

        fb=lb**2/(2*np.pi)
        
        for spec in spectra:
            
            Db_dict[ar1,ar2,spec,'auto']=np.mean(Db_dict[ar1,ar2,spec,'auto'],axis=0)
            Db_dict[ar1,ar2,spec,'cross']=np.mean(Db_dict[ar1,ar2,spec,'cross'],axis=0)
            nb_dict[ar1,ar2,spec]= (Db_dict[ar1,ar2,spec,'auto']-Db_dict[ar1,ar2,spec,'cross'])/2
            
            if (spec=='TT' or spec=='EE' or spec=='BB') & (ar1==ar2) :
    
                Nl = scipy.interpolate.interp1d(lb,nb_dict[ar1,ar2,spec], fill_value='extrapolate')
                Nl_interpolate[spec]=np.array([Nl(i) for i in lth])
                Nl_interpolate[spec]*=bl1[:len(lth)]*bl2[:len(lth)]
            
                plt.semilogy()
                plt.plot(lth,nlth[ar1,ar2,spec])
                plt.plot(lth,Nl_interpolate[spec])
                plt.savefig('%s/noise_interpolate_%s_%s_%s.png'%(ps_model_dir,ar1,ar2,spec),bbox_inches='tight')
                plt.clf()
                plt.close()

            else:
                Nl_interpolate[spec]=np.zeros(len(lth))
            
      

            plt.figure(figsize=(10,7))
            if spec=='TT' or spec=='EE' or spec=='BB':
                plt.semilogy()

            plt.ylabel('%s %sx%s'%(spec,ar1,ar2),fontsize=22)
            plt.plot(lth,Dlth[spec])
            plt.errorbar(lb,Db_dict[ar1,ar2,spec,'cross']*fb,fmt='.')
            plt.errorbar(lb,nb_dict[ar1,ar2,spec]*fb,fmt='.',color='red')
            plt.ylim(ylim[spec][0],ylim[spec][1])
            plt.savefig('%s/spectra_%s_%s_%s.png'%(ps_model_dir,ar1,ar2,spec),bbox_inches='tight')
            plt.clf()
            plt.close()
    

        np.savetxt('%s/noise_T_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,ar1,experiment,ar2),  np.transpose([lth,Nl_interpolate['TT']]))
        np.savetxt('%s/noise_P_%s_%sx%s_%s.dat'%(ps_model_dir,experiment,ar1,experiment,ar2),  np.transpose([lth,Nl_interpolate['EE']]))

        spec_name_noise='%s_%sx%s_%s_noise'%(experiment,ar1,experiment,ar2)
        so_spectra.write_ps(ps_model_dir+'/%s.dat'%spec_name_noise,lth,Nl_interpolate,type,spectra=spectra)



