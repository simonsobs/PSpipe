
import  numpy as np,pylab as plt
import os,sys
from pspy import so_cov
from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra,so_cov
import cov_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
experiment=d['experiment']
binning_file=d['binning_file']
lmax=d['lmax']
type=d['type']

new_lmin=50
new_lmax=lmax

iSim=np.arange(100)

bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
id=np.where((bin_c>new_lmin) & (bin_c<new_lmax))
n_bins= len(bin_hi[id])


like_dir='like_products'
mcm_dir='mcm'
cov_dir='covariance'


pspy_utils.create_directory(like_dir)

spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

g = open('%s/spectra_list.txt'%like_dir,mode="w")

spec_name_list=[]
for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue
                
                spec_name='%s_%sx%s_%s'%(exp1,f1,exp2,f2)
                
                for iii in iSim:
                    spec_name_cross='spectra/%s_%s_cross_%05d.dat'%(type,spec_name,iii)
                    l,ps=so_spectra.read_ps(spec_name_cross,spectra=spectra)
                    
                    l=l[id]
                    for spec in spectra:
                        ps[spec]=ps[spec][id]
                    
                    so_spectra.write_ps('%s/%s_%s_%05d.dat'%(like_dir,type,spec_name,iii),l,ps,type,spectra=spectra)
            
                spec_name_list+=[spec_name]
                
                prefix= '%s/%s'%(mcm_dir,spec_name)
                
                mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)
                Bbl_TT=Bbl['spin0xspin0']
                Bbl_TE=Bbl['spin0xspin2']
                Bbl_EE=Bbl['spin2xspin2'][:Bbl_TE.shape[0],:Bbl_TE.shape[1]]
                
                np.savetxt('%s/Bbl_%s_TT.dat'%(like_dir,spec_name),Bbl_TT[id[0],:])
                np.savetxt('%s/Bbl_%s_TE.dat'%(like_dir,spec_name),Bbl_TE[id[0],:])
                np.savetxt('%s/Bbl_%s_EE.dat'%(like_dir,spec_name),Bbl_EE[id[0],:])
        
            g.write('%s\n'%(spec_name))

g.close()


cov=np.load('monteCarlo/cov_restricted_all_cross.npy')

diagonal=True
boost_factor_diag=0.02
boost_factor_pseudo_diag=0.0

### construct analytic covariance from blocks ###
cov_analytic=cov_utils.blocks_to_cov_TTTEEE(cov_dir,spec_name_list,binning_file,lmax,boost_factor_diag=boost_factor_diag,boost_factor_pseudo_diag=boost_factor_pseudo_diag,diagonal=diagonal)

### select covariance ###
cov_analytic=cov_utils.select_cov_TTTEEE(cov_analytic,spec_name_list,binning_file,lmax,new_lmin=new_lmin,new_lmax=new_lmax)
cov=cov_utils.select_cov_TTTEEE(cov,spec_name_list,binning_file,lmax,new_lmin=new_lmin,new_lmax=new_lmax)

### check that the matrix is positive definite ###
print ('analytic cov is positive definite:', cov_utils.is_pos_def(cov_analytic))
print ('MC cov is positive definite:', cov_utils.is_pos_def(cov))

np.savetxt('%s/covariance.dat'%like_dir,cov_analytic)

os.system('cp %s %s/binning.dat'%(binning_file,like_dir))

