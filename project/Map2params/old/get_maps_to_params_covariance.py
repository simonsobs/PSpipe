from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra,so_cov
import  numpy as np, pylab as plt, healpy as hp
import os,sys
import so_noise_calculator_public_20180822 as noise_calc
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir='window'
mcm_dir='mcm'
cov_dir='covariance'
specDir='spectra'

experiment=d['experiment']
specDir='spectra'
clfile=d['clfile']
lmax=d['lmax']
type=d['type']
niter=d['niter']
binning_file=d['binning_file']
iStart= d['iStart']
iStop= d['iStop']
type=d['type']
lcut=d['lcut']
hdf5=d['hdf5']

include_fg=d['include_fg']
fg_dir=d['fg_dir']
fg_components=d['fg_components']



pspy_utils.create_directory(cov_dir)

if hdf5:
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


for exp in experiment:
    ns[exp]=d['nSplits_%s'%exp]

for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue

                l,bl1= np.loadtxt('beam/beam_%s_%s.dat'%(exp1,f1),unpack=True)
                l,bl2= np.loadtxt('beam/beam_%s_%s.dat'%(exp2,f2),unpack=True)
                bl1,bl2= bl1[:lmax],bl2[:lmax]

                for spec in ['TT','TE','ET','EE']:
                    
                    Dl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=bl1*bl2*Dlth[spec]
                    
                    if spec=='TT':
                        if include_fg:
                            flth_all=0
                            for foreground in fg_components:
                                l,flth=np.loadtxt('%s/tt_%s_%sx%s.dat'%(fg_dir,foreground,f1,f2),unpack=True)
                                flth_all+=flth[:lmax]
                            Dl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=bl1*bl2*(Dlth[spec]+flth_all)
                
                    if exp1==exp2:
                        
                        l,Nl_T=np.loadtxt('noise_ps/noise_T_%s_%sx%s_%s.dat'%(exp1,f1,exp2,f2),unpack=True)
                        l,Nl_P=np.loadtxt('noise_ps/noise_P_%s_%sx%s_%s.dat'%(exp1,f1,exp2,f2),unpack=True)

                        l,Nl_T,Nl_P=l[:lmax],Nl_T[:lmax],Nl_P[:lmax]
                        Nl_T[:lcut],Nl_P[:lcut]=0,0
            
                        if spec=='TT':
                            DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=Nl_T*l*(l+1)/(2*np.pi)*ns[exp1]
                        if spec=='EE':
                            DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=Nl_P*l*(l+1)/(2*np.pi)*ns[exp1]
                        if spec=='TE' or spec=='ET':
                            DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=Nl_T*0
                    else:
                        DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=np.zeros(lmax)
    
                    Dl_all['%s_%s'%(exp2,f2),'%s_%s'%(exp1,f1),spec]=Dl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]
                    DNl_all['%s_%s'%(exp2,f2),'%s_%s'%(exp1,f1),spec]=DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]

                spec_name+=['%s_%sx%s_%s'%(exp1,f1,exp2,f2)]

analytic_cov={}
cov={}
n1_list=[]
n2_list=[]
n3_list=[]
n4_list=[]
n_cov_element=0

for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        print (spec1,spec2)
        n1,n2=spec1.split('x')
        n3,n4=spec2.split('x')
        n1_list+=[n1]
        n2_list+=[n2]
        n3_list+=[n3]
        n4_list+=[n4]
        n_cov_element+=1

print (n_cov_element)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_cov_element-1)
print (subtasks)

for task in subtasks:
    print (task)
    
    n1=n1_list[int(task)]
    n2=n2_list[int(task)]
    n3=n3_list[int(task)]
    n4=n4_list[int(task)]
        
    prefix_ab= '%s/%sx%s'%(mcm_dir,n1,n2)
    prefix_cd= '%s/%sx%s'%(mcm_dir,n3,n4)

    mbb_inv_ab,Bbl_ab=so_mcm.read_coupling(prefix=prefix_ab,spin_pairs=spin_pairs)
    mbb_inv_ab=so_cov.extract_TTTEEE_mbb(mbb_inv_ab)
    mbb_inv_cd,Bbl_cd=so_mcm.read_coupling(prefix=prefix_cd,spin_pairs=spin_pairs)
    mbb_inv_cd=so_cov.extract_TTTEEE_mbb(mbb_inv_cd)

    win={}
    win['Ta']=so_map.read_map('%s/window_%s.fits'%(window_dir,n1))
    win['Tb']=so_map.read_map('%s/window_%s.fits'%(window_dir,n2))
    win['Tc']=so_map.read_map('%s/window_%s.fits'%(window_dir,n3))
    win['Td']=so_map.read_map('%s/window_%s.fits'%(window_dir,n4))
    win['Pa']=so_map.read_map('%s/window_%s.fits'%(window_dir,n1))
    win['Pb']=so_map.read_map('%s/window_%s.fits'%(window_dir,n2))
    win['Pc']=so_map.read_map('%s/window_%s.fits'%(window_dir,n3))
    win['Pd']=so_map.read_map('%s/window_%s.fits'%(window_dir,n4))


    coupling_dict=so_cov.cov_coupling_spin0and2(win, lmax, niter=niter)

    analytic_cov[n1,n2,n3,n4]=np.zeros((3*n_bins,3*n_bins))

    analytic_cov[n1,n2,n3,n4][:n_bins,:n_bins]=so_cov.bin_mat(coupling_dict['TaTcTbTd']*so_cov.chi(n1,n3,n2,n4,ns,l,Dl_all,DNl_all,'TTTT'),binning_file,lmax)
    analytic_cov[n1,n2,n3,n4][:n_bins,:n_bins]+=so_cov.bin_mat(coupling_dict['TaTdTbTc']*so_cov.chi(n1,n4,n2,n3,ns,l,Dl_all,DNl_all,'TTTT'),binning_file,lmax)
           
    analytic_cov[n1,n2,n3,n4][n_bins:2*n_bins,n_bins:2*n_bins]=so_cov.bin_mat(coupling_dict['TaTcPbPd']*so_cov.chi(n1,n3,n2,n4,ns,l,Dl_all,DNl_all,'TETE'),binning_file,lmax)
    analytic_cov[n1,n2,n3,n4][n_bins:2*n_bins,n_bins:2*n_bins]+=so_cov.bin_mat(coupling_dict['TaPdPbTc']*so_cov.chi(n1,n4,n2,n3,ns,l,Dl_all,DNl_all,'TTEE'),binning_file,lmax)

    analytic_cov[n1,n2,n3,n4][2*n_bins:3*n_bins,2*n_bins:3*n_bins]=so_cov.bin_mat(coupling_dict['PaPcPbPd']*so_cov.chi(n1,n3,n2,n4,ns,l,Dl_all,DNl_all,'EEEE'),binning_file,lmax)
    analytic_cov[n1,n2,n3,n4][2*n_bins:3*n_bins,2*n_bins:3*n_bins]+=so_cov.bin_mat(coupling_dict['PaPdPbPc']*so_cov.chi(n1,n4,n2,n3,ns,l,Dl_all,DNl_all,'EEEE'),binning_file,lmax)

    analytic_cov[n1,n2,n3,n4][n_bins:2*n_bins,:n_bins]=so_cov.bin_mat(coupling_dict['TaTcTbPd']*so_cov.chi(n1,n3,n2,n4,ns,l,Dl_all,DNl_all,'TTTE'),binning_file,lmax)
    analytic_cov[n1,n2,n3,n4][n_bins:2*n_bins,:n_bins]+=so_cov.bin_mat(coupling_dict['TaPdTbTc']*so_cov.chi(n1,n4,n2,n3,ns,l,Dl_all,DNl_all,'TTET'),binning_file,lmax)

    analytic_cov[n1,n2,n3,n4][2*n_bins:3*n_bins,:n_bins]=so_cov.bin_mat(coupling_dict['TaPcTbPd']*so_cov.chi(n1,n3,n2,n4,ns,l,Dl_all,DNl_all,'TTEE'),binning_file,lmax)
    analytic_cov[n1,n2,n3,n4][2*n_bins:3*n_bins,:n_bins]+=so_cov.bin_mat(coupling_dict['TaPdTbPc']*so_cov.chi(n1,n4,n2,n3,ns,l,Dl_all,DNl_all,'TTEE'),binning_file,lmax)

    analytic_cov[n1,n2,n3,n4][2*n_bins:3*n_bins,n_bins:2*n_bins]=so_cov.bin_mat(coupling_dict['PaTcPbPd']*so_cov.chi(n1,n3,n2,n4,ns,l,Dl_all,DNl_all,'EETE'),binning_file,lmax)
    analytic_cov[n1,n2,n3,n4][2*n_bins:3*n_bins,n_bins:2*n_bins]+=so_cov.bin_mat(coupling_dict['TaPdTbPc']*so_cov.chi(n1,n4,n2,n3,ns,l,Dl_all,DNl_all,'EEET'),binning_file,lmax)

    analytic_cov[n1,n2,n3,n4] = np.tril(analytic_cov[n1,n2,n3,n4]) + np.triu(analytic_cov[n1,n2,n3,n4].T, 1)
    analytic_cov[n1,n2,n3,n4]=np.dot(np.dot(mbb_inv_ab,analytic_cov[n1,n2,n3,n4]),mbb_inv_cd.T)

    Db_list1=[]
    Db_list2=[]

    for iii in range(iStart,iStop):
        spec_name_cross_1='%s_%sx%s_cross_%05d'%(type,n1,n2,iii)
        spec_name_cross_2='%s_%sx%s_cross_%05d'%(type,n3,n4,iii)
        if hdf5:
            lb,Db1=so_spectra.read_ps_hdf5(spectra_hdf5,spec_name_cross_1,spectra=spectra)
            lb,Db2=so_spectra.read_ps_hdf5(spectra_hdf5,spec_name_cross_2,spectra=spectra)
        else:
            lb,Db1=so_spectra.read_ps(specDir+'/%s.dat'%spec_name_cross_1,spectra=spectra)
            lb,Db2=so_spectra.read_ps(specDir+'/%s.dat'%spec_name_cross_2,spectra=spectra)

        vec1=[]
        vec2=[]
        for spec in ['TT','TE','EE']:
            vec1=np.append(vec1,Db1[spec])
            vec2=np.append(vec2,Db2[spec])
            
        Db_list1+=[vec1]
        Db_list2+=[vec2]

    cov[n1,n2,n3,n4]=0

    for iii in range(iStart,iStop):
        cov[n1,n2,n3,n4]+=np.outer(Db_list1[iii],Db_list2[iii])
    cov[n1,n2,n3,n4]= cov[n1,n2,n3,n4]/(iStop-iStart)-np.outer(np.mean(Db_list1,axis=0), np.mean(Db_list2,axis=0))

    np.save('%s/analytic_cov_%sx%s_%sx%s.npy'%(cov_dir,n1,n2,n3,n4), analytic_cov[n1,n2,n3,n4] )
    np.save('%s/mc_cov_%sx%s_%sx%s.npy'%(cov_dir,n1,n2,n3,n4), cov[n1,n2,n3,n4] )






