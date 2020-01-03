from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra,so_cov
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import h5py

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir='window'
mcm_dir='mcm'
cov_dir='covariance'
specDir='spectra'
ps_model_dir='model'

experiment=d['experiment']
lmax=d['lmax']
type=d['type']
niter=d['niter']
binning_file=d['binning_file']
run_name=d['run_name']

pspy_utils.create_directory(cov_dir)

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
n_bins=len(bin_hi)


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
                
                spec_name_combined='%s_%s_%sx%s_%s_cross'%(type,exp1,f1,exp2,f2)
                l,Dl=so_spectra.read_ps(ps_model_dir+'/%s.dat'%spec_name_combined,spectra=spectra)
                spec_name_noise='%s_%s_%sx%s_%s_noise'%(type,exp1,f1,exp2,f2)
                l,Nl=so_spectra.read_ps(ps_model_dir+'/%s.dat'%spec_name_noise,spectra=spectra)

                for spec in ['TT','TE','ET','EE']:
                    
                    Dl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=bl1*bl2*Dl[spec]
                    
                    if exp1==exp2:
                        DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=Nl[spec]*ns[exp1]
                    else:
                        DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]=np.zeros(lmax)
                    
    
                    Dl_all['%s_%s'%(exp2,f2),'%s_%s'%(exp1,f1),spec]=Dl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]
                    DNl_all['%s_%s'%(exp2,f2),'%s_%s'%(exp1,f1),spec]=DNl_all['%s_%s'%(exp1,f1),'%s_%s'%(exp2,f2),spec]

                spec_name+=['%s_%sx%s_%s'%(exp1,f1,exp2,f2)]

analytic_cov={}
cov={}
for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        
        print (spec1,spec2)
        n1,n2=spec1.split('x')
        n3,n4=spec2.split('x')
        
        prefix_ab= '%s/%sx%s'%(mcm_dir,n1,n2)
        prefix_cd= '%s/%sx%s'%(mcm_dir,n3,n4)

        mbb_inv_ab,Bbl_ab=so_mcm.read_coupling(prefix=prefix_ab,spin_pairs=spin_pairs)
        mbb_inv_ab=so_cov.extract_TTTEEE_mbb(mbb_inv_ab)
        mbb_inv_cd,Bbl_cd=so_mcm.read_coupling(prefix=prefix_cd,spin_pairs=spin_pairs)
        mbb_inv_cd=so_cov.extract_TTTEEE_mbb(mbb_inv_cd)

        win={}
        win['Ta']=so_map.read_map('%s/window_T_%s.fits'%(window_dir,n1))
        win['Tb']=so_map.read_map('%s/window_T_%s.fits'%(window_dir,n2))
        win['Tc']=so_map.read_map('%s/window_T_%s.fits'%(window_dir,n3))
        win['Td']=so_map.read_map('%s/window_T_%s.fits'%(window_dir,n4))
        win['Pa']=so_map.read_map('%s/window_P_%s.fits'%(window_dir,n1))
        win['Pb']=so_map.read_map('%s/window_P_%s.fits'%(window_dir,n2))
        win['Pc']=so_map.read_map('%s/window_P_%s.fits'%(window_dir,n3))
        win['Pd']=so_map.read_map('%s/window_P_%s.fits'%(window_dir,n4))


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


        np.save('%s/analytic_cov_%s_%sx%s_%sx%s.npy'%(cov_dir,run_name,n1,n2,n3,n4), analytic_cov[n1,n2,n3,n4] )



