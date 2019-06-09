#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir='window'
pspy_utils.create_directory(window_dir)
mcm_dir='mcm'
pspy_utils.create_directory(mcm_dir)
plot_dir='plot'
pspy_utils.create_directory(plot_dir)


experiment=d['experiment']
lmax=d['lmax']
ncomp=3
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']


for exp in experiment:
    
    freqs=d['freq_%s'%exp]
    
    for f in freqs:
        
        print ('mask_T_%s_%s'%(exp,f))
        
        mask_T=so_map.read_map(d['mask_T_%s_%s'%(exp,f)])
        mask_P=so_map.read_map(d['mask_P_%s_%s'%(exp,f)])
        survey_mask=so_map.read_map(d['survey_mask_%s_%s'%(exp,f)])
        
        mask_T.data*=survey_mask.data
        mask_P.data*=survey_mask.data
        
        
        window_T=so_window.create_apodization(mask_T, apo_type=d['apo_type_survey_%s'%exp], apo_radius_degree=d['apo_radius_survey_%s'%exp])
        window_T.write_map('%s/window_T_%s_%s.fits'%(window_dir,exp,f))
        window_T.plot(file_name='%s/window_T_%s_%s'%(plot_dir,exp,f))

        window_P=so_window.create_apodization(mask_P, apo_type=d['apo_type_survey_%s'%exp], apo_radius_degree=d['apo_radius_survey_%s'%exp])
        window_P.write_map('%s/window_P_%s_%s.fits'%(window_dir,exp,f))
        window_P.plot(file_name='%s/window_P_%s_%s'%(plot_dir,exp,f))

        del mask_T,mask_P,survey_mask,window_T,window_P


for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        l,bl1= np.loadtxt(d['beam_%s_%s'%(exp1,f1)],unpack=True)
        
        window1_T=so_map.read_map('%s/window_T_%s_%s.fits'%(window_dir,exp1,f1))
        window1_P=so_map.read_map('%s/window_P_%s_%s.fits'%(window_dir,exp1,f1))

        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue
                
                l,bl2= np.loadtxt(d['beam_%s_%s'%(exp2,f2)],unpack=True)

                window2_T=so_map.read_map('%s/window_T_%s_%s.fits'%(window_dir,exp2,f2))
                window2_P=so_map.read_map('%s/window_P_%s_%s.fits'%(window_dir,exp2,f2))

                print (exp1,f1,exp2,f2)
                mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(win1=(window1_T,window1_P),win2=(window2_T,window2_P),bl1=(bl1,bl1),bl2=(bl2,bl2),binning_file= d['binning_file'],niter=0, lmax=d['lmax'], type=d['type'],save_file='%s/%s_%sx%s_%s'%(mcm_dir,exp1,f1,exp2,f2))
