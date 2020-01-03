'''
This script analyze the result of the spectra simulation
It computes mean spectra and covariance from the sims
'''

import numpy as np
import pylab as plt
from pspy import so_dict,so_spectra,pspy_utils
import os,sys
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
binning_file=d['binning_file']
iStart=d['iStart']
iStop=d['iStop']
include_sys=d['include_systematics']
freqs=d['freqs']

if include_sys==True:
    simSpectraDir='sim_spectra_syst'
    mc_dir='monteCarlo_syst'
else:
    
    if d['use_ffp10']==True:
        simSpectraDir='sim_spectra_ffp10'
        mc_dir='monteCarlo_ffp10'
    else:
        simSpectraDir='sim_spectra'
        mc_dir='monteCarlo'


pspy_utils.create_directory(mc_dir)


freq_pairs=[]
for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        freq_pairs+=[[freq1,freq2]]

print (freq_pairs)

halfmission_pairs=[['hm1','hm1'],['hm1','hm2'],['hm2','hm2']]

for hm_pair in halfmission_pairs:
    hm0,hm1=hm_pair
    hmname='%sx%s'%(hm0,hm1)
    
    spec_name=[]
    vec_lb=[]
    id_start={}
    id_stop={}
    spec_name_list=[]
    count=0
    
    vec=[]
    for fpair in freq_pairs:
        f0,f1=fpair
        fname='%sx%s'%(f0,f1)
        spec_name='Planck_%sxPlanck_%s-%s'%(f0,f1,hmname)

        lb,ps_dict=so_spectra.read_ps('%s/sim_spectra_%s_%04d.dat'%(simSpectraDir,spec_name,000),spectra=spectra)
        lmin,lmax=d['lrange_%sx%s'%(f0,f1)]

        id=np.where((lb>lmin) &(lb<lmax))
            
        lb=lb[id]
        for id_spec,spec in enumerate(['TT','TE','EE','r']):
            
            if spec !='r':
                ps_dict[spec]=ps_dict[spec][id]
                vec=np.append(vec,ps_dict[spec])
            else:
                r=ps_dict['TE']/np.sqrt(ps_dict['TT']*ps_dict['EE'])
                vec=np.append(vec,r)
    
            spec_name='%s_%s_%s'%(spec,fname,hmname)
            id_start[spec_name]=count
            id_stop[spec_name]=count+len(lb)
            spec_name_list+=[spec_name]
            vec_lb=np.append(vec_lb,lb)
            count+=len(lb)


    nbins_tot=len(vec)
    vec_mean=np.zeros(nbins_tot)
    cov_mean=np.zeros((nbins_tot,nbins_tot))

    
    for iii in range(iStart,iStop):

        vec=[]

        for fpair in freq_pairs:
            
            f0,f1=fpair
            
            fname='%sx%s'%(f0,f1)
        
            spec_name='Planck_%sxPlanck_%s-%s'%(f0,f1,hmname)
        
            lb,ps_dict=so_spectra.read_ps('%s/sim_spectra_%s_%04d.dat'%(simSpectraDir,spec_name,iii),spectra=spectra)
        
            ps_dict['TE']=(ps_dict['TE']+ps_dict['ET'])/2
        
            if (f0 !=f1) & (hm0 !=hm1):
                
                spec_name='Planck_%sxPlanck_%s-%sx%s'%(f0,f1,hm1,hm0)
                lb,ps_dict_mirror=so_spectra.read_ps('%s/sim_spectra_%s_%04d.dat'%(simSpectraDir,spec_name,iii),spectra=spectra)
                ps_dict_mirror['TE']=(ps_dict_mirror['TE']+ps_dict_mirror['ET'])/2
            
                ps_dict['TE']=(ps_dict['TE']+ps_dict_mirror['TE'])/2

            lmin,lmax=d['lrange_%sx%s'%(f0,f1)]

            id=np.where((lb>lmin) &(lb<lmax))
        

            for spec in ['TT','TE','EE','r']:
            
                if spec !='r':
                    ps_dict[spec]=ps_dict[spec][id]
                    vec=np.append(vec,ps_dict[spec])
                else:
                    r=ps_dict['TE']/np.sqrt(ps_dict['TT']*ps_dict['EE'])
                    vec=np.append(vec,r)

        vec_mean+=vec
        cov_mean+=np.outer(vec,vec)

    vec_mean/=(iStop-iStart)
    cov_mean=cov_mean/(iStop-iStart)-np.outer(vec_mean,vec_mean)

    np.savetxt('%s/full_cov_mat_%s.dat'%(mc_dir,hmname),cov_mean )

    for spec_name1 in spec_name_list:
        #print (vec_lb.shape,vec_mean.shape,cov_mean.shape)

        lb=vec_lb[id_start[spec_name1]:id_stop[spec_name1]]
        ps=vec_mean[id_start[spec_name1]:id_stop[spec_name1]]
        cov=cov_mean[id_start[spec_name1]:id_stop[spec_name1],id_start[spec_name1]:id_stop[spec_name1]]
        error=np.sqrt(cov.diagonal())
        
        print (spec_name1,lb.shape,ps.shape,error.shape)
    
        np.savetxt('%s/spectra_%s.dat'%(mc_dir,spec_name1), np.transpose([lb,ps,error]))
    
        for spec_name2 in spec_name_list:
            cov_select=cov_mean[id_start[spec_name1]:id_stop[spec_name1],id_start[spec_name2]:id_stop[spec_name2]]
            np.savetxt('%s/select_cov_mat_%s_%s.dat'%(mc_dir,spec_name1,spec_name2),cov_select)
            np.savetxt('%s/diagonal_select_cov_mat_%s_%s.dat'%(mc_dir,spec_name1,spec_name2),cov_select.diagonal())









