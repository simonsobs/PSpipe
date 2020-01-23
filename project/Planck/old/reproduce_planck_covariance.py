from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra,so_cov
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import h5py

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir='window'
mcm_dir='mcm'
specDir='spectra'
ps_model_dir='model'
cov_dir='covariance'

pspy_utils.create_directory(cov_dir)


freqs=d['freqs']
lmax=d['lmax']
type=d['type']
niter=d['niter']
binning_file=d['binning_file']

experiment='Planck'

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
n_bins=len(bin_hi)


Dl_all={}
DNl_all={}
ns={}
bl1,bl2={},{}
spec_name=[]

ns['%s'%(experiment)]=2


for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        
        print ('beam_%s_hm1'%freq1)
        
        l,bl1_hm1_T= np.loadtxt(d['beam_%s_hm1_T'%freq1],unpack=True)
        l,bl1_hm2_T= np.loadtxt(d['beam_%s_hm2_T'%freq1],unpack=True)
        l,bl1_hm1_pol= np.loadtxt(d['beam_%s_hm1_pol'%freq1],unpack=True)
        l,bl1_hm2_pol= np.loadtxt(d['beam_%s_hm2_pol'%freq1],unpack=True)

        l,bl2_hm1_T= np.loadtxt(d['beam_%s_hm1_T'%freq2],unpack=True)
        l,bl2_hm2_T= np.loadtxt(d['beam_%s_hm2_T'%freq2],unpack=True)
        l,bl2_hm1_pol= np.loadtxt(d['beam_%s_hm1_pol'%freq2],unpack=True)
        l,bl2_hm2_pol= np.loadtxt(d['beam_%s_hm2_pol'%freq2],unpack=True)

        bl1_hm1_T,bl1_hm2_T,bl2_hm1_T,bl2_hm2_T= bl1_hm1_T[:lmax],bl1_hm2_T[:lmax],bl2_hm1_T[:lmax],bl2_hm2_T[:lmax]
        bl1_hm1_pol,bl1_hm2_pol,bl2_hm1_pol,bl2_hm2_pol= bl1_hm1_pol[:lmax],bl1_hm2_pol[:lmax],bl2_hm1_pol[:lmax],bl2_hm2_pol[:lmax]

        bl1['TT']=np.sqrt(bl1_hm1_T*bl1_hm2_T)
        bl2['TT']=np.sqrt(bl2_hm1_T*bl2_hm2_T)

        bl1['EE']=np.sqrt(bl1_hm1_pol*bl1_hm2_pol)
        bl2['EE']=np.sqrt(bl2_hm1_pol*bl2_hm2_pol)

        bl1['TE']=np.sqrt(bl1['EE']*bl1['TT'])
        bl2['TE']=np.sqrt(bl2['EE']*bl2['TT'])
        
        bl1['ET']=bl1['TE']
        bl2['ET']=bl2['TE']

        lth,ps_th=pspy_utils.ps_lensed_theory_to_dict(d['theoryfile'],output_type=type,lmax=lmax,lstart=2)
        
        spec_name_noise='mean_%s_%sx%s_%s_noise'%(experiment,freq1,experiment,freq2)
        l,Nl=so_spectra.read_ps(ps_model_dir+'/%s.dat'%spec_name_noise,spectra=spectra)
                
        for spec in ['TT','TE','ET','EE']:
                    
            Dl_all['%s_%s'%(experiment,freq1),'%s_%s'%(experiment,freq2),spec]=bl1[spec]*bl2[spec]*ps_th[spec]
                    
            if freq1==freq2:
                DNl_all['%s_%s'%(experiment,freq1),'%s_%s'%(experiment,freq2),spec]=Nl[spec]*ns['%s'%(experiment)]
            else:
                DNl_all['%s_%s'%(experiment,freq1),'%s_%s'%(experiment,freq2),spec]=np.zeros(lmax)
                    

            Dl_all['%s_%s'%(experiment,freq2),'%s_%s'%(experiment,freq1),spec]=Dl_all['%s_%s'%(experiment,freq1),'%s_%s'%(experiment,freq2),spec]
            DNl_all['%s_%s'%(experiment,freq2),'%s_%s'%(experiment,freq1),spec]=DNl_all['%s_%s'%(experiment,freq1),'%s_%s'%(experiment,freq2),spec]
                
        spec_name+=['%s_%sx%s_%s'%(experiment,freq1,experiment,freq2)]

analytic_cov={}
cov={}
for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        
        print (spec1,spec2)
        n1,n2=spec1.split('x')
        n3,n4=spec2.split('x')
        
        prefix_ab= '%s/%sx%s-hm1xhm2'%(mcm_dir,n1,n2)
        prefix_cd= '%s/%sx%s-hm1xhm2'%(mcm_dir,n1,n2)
        
        mbb_inv_ab,Bbl_ab=so_mcm.read_coupling(prefix=prefix_ab,spin_pairs=spin_pairs)
        mbb_inv_ab=so_cov.extract_TTTEEE_mbb(mbb_inv_ab)
        mbb_inv_cd,Bbl_cd=so_mcm.read_coupling(prefix=prefix_cd,spin_pairs=spin_pairs)
        mbb_inv_cd=so_cov.extract_TTTEEE_mbb(mbb_inv_cd)
        
        win={}
        win['Ta']=so_map.read_map('%s/window_T_%s-hm1.fits'%(window_dir,n1))
        win['Tb']=so_map.read_map('%s/window_T_%s-hm2.fits'%(window_dir,n2))
        win['Tc']=so_map.read_map('%s/window_T_%s-hm1.fits'%(window_dir,n3))
        win['Td']=so_map.read_map('%s/window_T_%s-hm2.fits'%(window_dir,n4))
        win['Pa']=so_map.read_map('%s/window_P_%s-hm1.fits'%(window_dir,n1))
        win['Pb']=so_map.read_map('%s/window_P_%s-hm2.fits'%(window_dir,n2))
        win['Pc']=so_map.read_map('%s/window_P_%s-hm1.fits'%(window_dir,n3))
        win['Pd']=so_map.read_map('%s/window_P_%s-hm2.fits'%(window_dir,n4))
        
        
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
        
        
        np.save('%s/analytic_cov_%sx%s_%sx%s.npy'%(cov_dir,n1,n2,n3,n4), analytic_cov[n1,n2,n3,n4] )


