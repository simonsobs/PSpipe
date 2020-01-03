import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap
import time,os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectraDir='spectra'
try:
    os.makedirs(spectraDir)
except:
    pass


spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

arrays=d['arrays']
niter=d['niter']
lmax=d['lmax']
type=d['type']
binning_file=d['binning_file']
theoryfile=d['theoryfile']
fsky={}
fsky['pa1']='fsky0.01081284'
fsky['pa2']='fsky0.01071187'

apo = so_map.read_map(d['apo_path'])
box=so_map.bounding_box_from_map(apo)

for ar in arrays:
    t=time.time()

    window=so_map.read_map(d['window_T_%s'%ar])
    window=so_map.get_submap_car(window,box,mode='round')
    
    window_tuple=(window,window)
    
    print ("compute mcm and Bbl ...")
    beam= np.loadtxt(d['beam_%s'%ar])
    l,bl=beam[:,0],beam[:,1]
    bl_tuple=(bl,bl)
    
    mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(window_tuple, binning_file,niter=niter, bl1=bl_tuple, lmax=lmax, type=type)
    
    almList=[]
    nameList=[]
    
    if d['use_filtered_maps']==True:
        map_T=d['map_T_%s_filtered'%ar]
        map_Q=d['map_Q_%s_filtered'%ar]
        map_U=d['map_U_%s_filtered'%ar]
    else:
        map_T=d['map_T_%s'%ar]
        map_Q=d['map_Q_%s'%ar]
        map_U=d['map_U_%s'%ar]

    print ("compute harmonic transform ...")
    count=0
    for T,Q,U in zip(map_T,map_Q,map_U):
        map=so_map.from_components(T,Q,U)
        map=so_map.get_submap_car(map,box,mode='floor')

        if d['use_filtered_maps']==False:
            map = so_map_preprocessing.get_map_kx_ky_filtered_pyfftw(map,apo,d['filter_dict'])
        
        almList+=[ sph_tools.get_alms(map,window_tuple,niter,lmax) ]
        nameList+=['split_%d_%s'%(count,ar)]
        count+=1

    print ("get spectra ...")

    Db_dict={}
    Db_dict_auto={}
    Db_dict_cross={}
    for s1 in spectra:
        Db_dict_auto[s1]=[]
        Db_dict_cross[s1]=[]

    spec_name_list=[]
    for name1, alm1, c1  in zip(nameList,almList,np.arange(count)):
        for name2, alm2, c2  in zip(nameList,almList,np.arange(count)):
            if c1>c2: continue
            l,ps= so_spectra.get_spectra(alm1,alm2,spectra=spectra)
            spec_name='%sx%s'%(name1,name2)
            lb,Db_dict[spec_name]=so_spectra.bin_spectra(l,ps,binning_file,lmax,type=type,mbb_inv=mbb_inv,spectra=spectra)
            spec_name_list+=[spec_name]
            so_spectra.write_ps('%s/spectra_%s.dat'%(spectraDir,spec_name),lb,Db_dict[spec_name],type=type,spectra=spectra)
            
            if c1==c2:
                print ('auto %dx%d'%(c1,c2))
                for s1 in spectra:
                    Db_dict_auto[s1]+=[Db_dict[spec_name][s1]]
            else:
                print ('cross %dx%d'%(c1,c2))
                for s1 in spectra:
                    Db_dict_cross[s1]+=[Db_dict[spec_name][s1]]

    for s1 in spectra:
        Db_dict_auto[s1]=np.mean(Db_dict_auto[s1],axis=0)
        Db_dict_cross[s1]=np.mean(Db_dict_cross[s1],axis=0)

    so_spectra.write_ps('%s/spectra_auto_%s.dat'%(spectraDir,ar),lb,Db_dict_auto,type=type,spectra=spectra)
    so_spectra.write_ps('%s/spectra_cross_%s.dat'%(spectraDir,ar),lb,Db_dict_cross,type=type,spectra=spectra)

    Db_dict_cross['TE']=(Db_dict_cross['TE']+Db_dict_cross['ET'])/2

    for s1 in ['TT','TE','EE']:
        lb_steve,Db_steve,sigmab_steve,Nb_steve=np.loadtxt('%s/deep56_s14_%s_f150_c7v5_car_190220_rect_window0_%s_lmax7925_%s_output.txt'%(d['steve_ps_dir'],ar,s1,fsky[ar]),unpack=True)

        new_lb=lb.copy()
        if len(lb)>len(lb_steve):
            new_lb,Db_dict_cross['%s'%s1]= lb[:len(lb_steve)],Db_dict_cross['%s'%s1][:len(lb_steve)]
        
        if len(lb_steve)>len(lb):
            Db_steve= Db_steve[:len(lb)]
            lb_steve=lb_steve[:len(lb)]
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        if s1=='TT':
            plt.semilogy()
        plt.plot(new_lb,Db_dict_cross['%s'%s1],'o',label='pspy')
        plt.plot(lb_steve,Db_steve,label='steve')
        plt.ylabel(r'$D^{%s}_{\ell}$'%s1,fontsize=20)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.subplot(2,1,2)
        plt.plot(new_lb,Db_dict_cross['%s'%s1]/Db_steve[:len(lb)],'o',label='pspy')
        plt.ylabel(r'$D^{%s, pspy}_{\ell}/D^{%s, Steve}_{\ell}$'%(s1,s1),fontsize=20)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.legend()
        plt.savefig('spectra/spectra_%s_%s.png'%(s1,ar))
        plt.clf()
        plt.close()

    print ('reproducing steve deep56 %s computation took: %.02f'%(ar,time.time()-t))
