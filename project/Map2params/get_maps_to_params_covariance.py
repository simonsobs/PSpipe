from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra,so_cov
import  numpy as np, pylab as plt, healpy as hp
import os,sys
import so_noise_calculator_public_20180822 as noise_calc
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

freqs=d['freqs']
window_dir='window'
specDir='spectra'
clfile=d['clfile']
lmax=d['lmax']
type=d['type']
ns=d['nSplits']
niter=d['niter']
binning_file=d['binning_file']
mcm_dir='mcm'
iStart= d['iStart']
iStop= d['iStop']
type=d['type']
lcut=d['lcut']
cov_dir='covariance'


pspy_utils.create_directory(cov_dir)

spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'r')


ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
n_bins=len(bin_hi)


lth,Dlth=pspy_utils.ps_lensed_theory_to_dict(clfile,output_type=type,lmax=lmax,lstart=2)


Dl_all={}
DNl_all={}
ns={}


spec_name=[]
for fid1,f1 in enumerate(freqs):
    ns[f1]=2

    for fid2,f2 in enumerate(freqs):
        if fid1>fid2: continue
        l,bl1= np.loadtxt('beam/beam_%s.dat'%f1,unpack=True)
        l,bl2= np.loadtxt('beam/beam_%s.dat'%f2,unpack=True)
        bl1,bl2= bl1[:lmax],bl2[:lmax]

        for spec in ['TT','TE','ET','EE']:
            
            l,Nl_T=np.loadtxt('noise_ps/noise_T_%sx%s.dat'%(f1,f2),unpack=True)
            l,Nl_P=np.loadtxt('noise_ps/noise_P_%sx%s.dat'%(f1,f2),unpack=True)

            l,Nl_T,Nl_P=l[:lmax],Nl_T[:lmax],Nl_P[:lmax]
            Nl_T[:lcut],Nl_P[:lcut]=0,0
            
            Dl_all[f1,f2,spec]=bl1*bl2*Dlth[spec]
            
            if spec=='TT':
                DNl_all[f1,f2,spec]=Nl_T*l*(l+1)/(2*np.pi)*ns[f1]
            if spec=='EE':
                DNl_all[f1,f2,spec]=Nl_P*l*(l+1)/(2*np.pi)*ns[f1]
            if spec=='TE' or spec=='ET':
                DNl_all[f1,f2,spec]=Nl_T*0

        spec_name+=['%sx%s'%(f1,f2)]



analytic_cov={}
cov={}
for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        f1,f2=spec1.split('x')
        f3,f4=spec2.split('x')


        prefix_ab= '%s/%sx%s'%(mcm_dir,f1,f2)
        prefix_cd= '%s/%sx%s'%(mcm_dir,f3,f4)

        mbb_inv_ab,Bbl_ab=so_mcm.read_coupling(prefix=prefix_ab,spin_pairs=spin_pairs)
        mbb_inv_cd,Bbl_cd=so_mcm.read_coupling(prefix=prefix_cd,spin_pairs=spin_pairs)

     
     #nl_th=pspy_utils.get_nlth_dict(10,type,lmax,spectra=spectra)
     #  survey_id= ['a','b','c','d']
     #  survey_name=['split_0','split_1','split_0','split_1']
     #  name_list=[]
     #  id_list=[]
     #  for field in ['T','E']:
     #      for s,id in zip(survey_name,survey_id):
     #          name_list+=['%s%s'%(field,s)]
     #          id_list+=['%s%s'%(field,id)]
     #  Clth_dict={}
     #  for name1,id1 in zip(name_list,id_list):
     #      for name2,id2 in zip(name_list,id_list):
     #          spec=id1[0]+id2[0]
     #          Clth_dict[id1+id2]=Dl_all[f1,f2,spec]+DNl_all[f1,f2,spec]*so_cov.delta2(name1,name2)

#        prefix= '%s/%sx%s'%(mcm_dir,f1,f2)
#       mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)
#       window=so_map.read_map('%s/window_%s.fits'%(window_dir,f1))
#       coupling_dict=so_cov.cov_coupling_spin0and2(window, lmax, niter=niter)
#       analytic_cov=so_cov.cov_spin0and2(Clth_dict,coupling_dict,binning_file,lmax,mbb_inv_ab,mbb_inv_cd)
#       analytic_corr=so_cov.cov2corr(analytic_cov)
#       plt.matshow(analytic_corr)
#       plt.show()




        win={}
        win['Ta']=so_map.read_map('%s/window_%s.fits'%(window_dir,f1))
        win['Tb']=so_map.read_map('%s/window_%s.fits'%(window_dir,f2))
        win['Tc']=so_map.read_map('%s/window_%s.fits'%(window_dir,f3))
        win['Td']=so_map.read_map('%s/window_%s.fits'%(window_dir,f4))
        win['Pa']=so_map.read_map('%s/window_%s.fits'%(window_dir,f1))
        win['Pb']=so_map.read_map('%s/window_%s.fits'%(window_dir,f2))
        win['Pc']=so_map.read_map('%s/window_%s.fits'%(window_dir,f3))
        win['Pd']=so_map.read_map('%s/window_%s.fits'%(window_dir,f4))

#        print (so_cov.bin_mat(coupling_dict['TaTcTbTd'],binning_file,lmax))


        coupling_dict=so_cov.cov_coupling_spin0and2(win, lmax, niter=niter)

#window=so_map.read_map('%s/window_%s.fits'%(window_dir,f1))
#       coupling_dict=so_cov.cov_coupling_spin0and2(window, lmax, niter=niter)

#        print (so_cov.bin_mat(coupling_dict['TaTcTbTd'],binning_file,lmax))
            
        analytic_cov[f1,f2,f3,f4]=np.zeros((3*n_bins,3*n_bins))

        analytic_cov[f1,f2,f3,f4][:n_bins,:n_bins]=so_cov.bin_mat(coupling_dict['TaTcTbTd']*so_cov.chi(f1,f3,f2,f4,ns,l,Dl_all,DNl_all,'TTTT'),binning_file,lmax)
        analytic_cov[f1,f2,f3,f4][:n_bins,:n_bins]+=so_cov.bin_mat(coupling_dict['TaTdTbTc']*so_cov.chi(f1,f4,f2,f3,ns,l,Dl_all,DNl_all,'TTTT'),binning_file,lmax)
           
        analytic_cov[f1,f2,f3,f4][n_bins:2*n_bins,n_bins:2*n_bins]=so_cov.bin_mat(coupling_dict['TaTcPbPd']*so_cov.chi(f1,f3,f2,f4,ns,l,Dl_all,DNl_all,'TETE'),binning_file,lmax)
        analytic_cov[f1,f2,f3,f4][n_bins:2*n_bins,n_bins:2*n_bins]+=so_cov.bin_mat(coupling_dict['TaPdPbTc']*so_cov.chi(f1,f4,f2,f3,ns,l,Dl_all,DNl_all,'TTEE'),binning_file,lmax)

        analytic_cov[f1,f2,f3,f4][2*n_bins:3*n_bins,2*n_bins:3*n_bins]=so_cov.bin_mat(coupling_dict['PaPcPbPd']*so_cov.chi(f1,f3,f2,f4,ns,l,Dl_all,DNl_all,'EEEE'),binning_file,lmax)
        analytic_cov[f1,f2,f3,f4][2*n_bins:3*n_bins,2*n_bins:3*n_bins]+=so_cov.bin_mat(coupling_dict['PaPdPbPc']*so_cov.chi(f1,f4,f2,f3,ns,l,Dl_all,DNl_all,'EEEE'),binning_file,lmax)

        analytic_cov[f1,f2,f3,f4][n_bins:2*n_bins,:n_bins]=so_cov.bin_mat(coupling_dict['TaTcTbPd']*so_cov.chi(f1,f3,f2,f4,ns,l,Dl_all,DNl_all,'TTTE'),binning_file,lmax)
        analytic_cov[f1,f2,f3,f4][n_bins:2*n_bins,:n_bins]+=so_cov.bin_mat(coupling_dict['TaPdTbTc']*so_cov.chi(f1,f4,f2,f3,ns,l,Dl_all,DNl_all,'TTET'),binning_file,lmax)

        analytic_cov[f1,f2,f3,f4][2*n_bins:3*n_bins,:n_bins]=so_cov.bin_mat(coupling_dict['TaPcTbPd']*so_cov.chi(f1,f3,f2,f4,ns,l,Dl_all,DNl_all,'TTEE'),binning_file,lmax)
        analytic_cov[f1,f2,f3,f4][2*n_bins:3*n_bins,:n_bins]+=so_cov.bin_mat(coupling_dict['TaPdTbPc']*so_cov.chi(f1,f4,f2,f3,ns,l,Dl_all,DNl_all,'TTEE'),binning_file,lmax)

        analytic_cov[f1,f2,f3,f4][2*n_bins:3*n_bins,n_bins:2*n_bins]=so_cov.bin_mat(coupling_dict['PaTcPbPd']*so_cov.chi(f1,f3,f2,f4,ns,l,Dl_all,DNl_all,'EETE'),binning_file,lmax)
        analytic_cov[f1,f2,f3,f4][2*n_bins:3*n_bins,n_bins:2*n_bins]+=so_cov.bin_mat(coupling_dict['TaPdTbPc']*so_cov.chi(f1,f4,f2,f3,ns,l,Dl_all,DNl_all,'EEET'),binning_file,lmax)

        analytic_cov[f1,f2,f3,f4] = np.tril(analytic_cov[f1,f2,f3,f4]) + np.triu(analytic_cov[f1,f2,f3,f4].T, 1)


        mbb_inv_ab=so_cov.extract_TTTEEE_mbb(mbb_inv_ab)
        mbb_inv_cd=so_cov.extract_TTTEEE_mbb(mbb_inv_cd)
    
        analytic_cov[f1,f2,f3,f4]=np.dot(np.dot(mbb_inv_ab,analytic_cov[f1,f2,f3,f4]),mbb_inv_cd.T)


        Db_list1=[]
        Db_list2=[]

        for iii in range(iStart,iStop):
            spec_name_cross_1='%s_%sx%s_cross_%05d'%(type,f1,f2,iii)
            spec_name_cross_2='%s_%sx%s_cross_%05d'%(type,f3,f4,iii)
            lb,Db1=so_spectra.read_ps_hdf5(spectra_hdf5,spec_name_cross_1,spectra=spectra)
            lb,Db2=so_spectra.read_ps_hdf5(spectra_hdf5,spec_name_cross_2,spectra=spectra)
            vec1=[]
            vec2=[]
            for spec in ['TT','TE','EE']:
                vec1=np.append(vec1,Db1[spec])
                vec2=np.append(vec2,Db2[spec])
            Db_list1+=[vec1]
            Db_list2+=[vec2]

        cov[f1,f2,f3,f4]=0

        for iii in range(iStart,iStop):
            
            cov[f1,f2,f3,f4]+=np.outer(Db_list1[iii],Db_list2[iii])
        cov[f1,f2,f3,f4]= cov[f1,f2,f3,f4]/(iStop-iStart)-np.outer(np.mean(Db_list1,axis=0), np.mean(Db_list2,axis=0))

        corr=so_cov.cov2corr(cov[f1,f2,f3,f4])
        analytic_corr=so_cov.cov2corr(analytic_cov[f1,f2,f3,f4])

        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        plt.title('Monte-Carlo correlation matrix',fontsize=22)
        plt.imshow(corr,vmin=-0.5,vmax=0.5,origin='lower')
        ticks=np.arange(4,3*n_bins,15)
        labels=(lb[ticks%n_bins]).astype(int)
        plt.xticks(ticks,labels)
        plt.yticks(ticks,labels)
        plt.ylabel(r'$\ell_{TT}  \hspace{3}   \ell_{TE} \hspace{3}   \ell_{EE}$',fontsize=22)
        plt.xlabel(r'$\ell_{TT}  \hspace{3}   \ell_{TE} \hspace{3}   \ell_{EE}$',fontsize=22)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.ylabel(r'$\ell_{TT}  \hspace{3}   \ell_{TE} \hspace{3}   \ell_{EE}$',fontsize=22)
        plt.xlabel(r'$\ell_{TT}  \hspace{3}   \ell_{TE} \hspace{3}   \ell_{EE}$',fontsize=22)
        plt.xticks(ticks,labels)
        plt.yticks(ticks,labels)
        plt.title('Analytic correlation matrix',fontsize=22)
        plt.imshow(analytic_corr,vmin=-0.5,vmax=0.5,origin='lower')
        plt.colorbar()
        
        #plt.show()

        plt.savefig('%s/correlation_matrix_%sx%s_%sx%s.png'%(cov_dir,f1,f2,f3,f4),bbox_inches='tight')
        plt.clf()
        plt.close()
    
    
        plt.figure(figsize=(15,15))
        count=1
        for bl in ['TTTT','EEEE','TETE','TTEE','TEEE','TETT']:
            plt.subplot(2,3,count)
            cov_select=so_cov.selectblock(cov[f1,f2,f3,f4], ['TT','TE','EE'],n_bins,block=bl)
            analytic_cov_select=so_cov.selectblock(analytic_cov[f1,f2,f3,f4],  ['TT','TE','EE'],n_bins,block=bl)
            var = cov_select.diagonal()
            analytic_var = analytic_cov_select.diagonal()
            if count==1:
                plt.semilogy()
            plt.plot(lb[1:],var[1:],'o',label='MC %sx%s'%(bl[:2],bl[2:4]))
            plt.plot(lb[1:],analytic_var[1:],label='Analytic %sx%s'%(bl[:2],bl[2:4]))
            if count==1 or count==4:
                plt.ylabel(r'$\sigma^{2}_{\ell}$',fontsize=22)
            if count >3:
                plt.xlabel(r'$\ell$',fontsize=22)
            plt.legend()
            count+=1
    
#plt.show()
    
        plt.savefig('%s/cov_element_comparison_%sx%s_%sx%s.png'%(cov_dir,f1,f2,f3,f4),bbox_inches='tight')
        plt.clf()
        plt.close()








