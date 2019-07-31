from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
from matplotlib.pyplot import cm
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import h5py

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

iStart=d['iStart']
iStop=d['iStop']
lmax=d['lmax']
type=d['type']
clfile=d['clfile']
arrays=d['arrays']
experiment='Planck'
split=['hm1','hm2']


specDir='sim_spectra'

mcm_dir='mcm'
mc_dir='monteCarlo'

lth,ps_th=pspy_utils.ps_lensed_theory_to_dict(d['theoryfile'],output_type=type,lmax=lmax,lstart=2)


pspy_utils.create_directory(mc_dir)

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']


vec_list=[]
vec_list_restricted=[]
for iii in range(iStart,iStop):
    vec=[]
    vec_restricted=[]
    count=0
    for spec in spectra:
        for c1,ar1 in enumerate(arrays):
            for c2,ar2 in enumerate(arrays):
                if c1>c2: continue
                for s1,hm1 in enumerate(split):
                    for s2,hm2 in enumerate(split):
                        if (s1>s2) & (c1==c2): continue

                        spec_name='%s_%sx%s_%s-%sx%s'%(experiment,ar1,experiment,ar2,hm1,hm2)

                        lb,Db=so_spectra.read_ps(specDir+'/sim_spectra_%s_%04d.dat'%(spec_name,iii),spectra=spectra)

                        n_bins=len(lb)
                        vec=np.append(vec,Db[spec])
                        if spec=='TT' or spec=='EE':
                            vec_restricted=np.append(vec_restricted,Db[spec])
                        if spec=='TE':
                            vec_restricted=np.append(vec_restricted,(Db['TE']+Db['ET'])/2)

                                
    vec_list+=[vec]
    vec_list_restricted+=[vec_restricted]

mean_vec=np.mean(vec_list,axis=0)
mean_vec_restricted=np.mean(vec_list_restricted,axis=0)

cov=0
cov_restricted=0

for iii in range(iStart,iStop):
    cov+=np.outer(vec_list[iii],vec_list[iii])
    cov_restricted+=np.outer(vec_list_restricted[iii],vec_list_restricted[iii])

cov=cov/(iStop-iStart)-np.outer(mean_vec, mean_vec)
cov_restricted=cov_restricted/(iStop-iStart)-np.outer(mean_vec_restricted, mean_vec_restricted)


np.save('%s/cov_all.npy'%(mc_dir),cov)
np.save('%s/cov_restricted_all.npy'%(mc_dir),cov_restricted)

id_spec=0
for spec in spectra:
    for c1,ar1 in enumerate(arrays):
        for c2,ar2 in enumerate(arrays):
            if c1>c2: continue
            for s1,hm1 in enumerate(split):
                for s2,hm2 in enumerate(split):
                    if (s1>s2) & (c1==c2): continue
                    
                    prefix= '%s/%s_%sx%s_%s-%sx%s'%(mcm_dir,experiment,ar1,experiment,ar2,hm1,hm2)
                    mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)
                    bin_theory=so_mcm.apply_Bbl(Bbl,ps_th,spectra=spectra)


                    spec_name='%s_%sx%s_%s-%sx%s'%(experiment,ar1,experiment,ar2,hm1,hm2)
                    
                    mean=mean_vec[id_spec*n_bins:(id_spec+1)*n_bins]
                    std=np.sqrt(cov[id_spec*n_bins:(id_spec+1)*n_bins,id_spec*n_bins:(id_spec+1)*n_bins].diagonal())
                        
                    np.savetxt('%s/mean_spectra_%s_%s.dat'%(mc_dir,spec,spec_name), np.array([lb,mean,std]).T)
                    id_spec+=1
                    
                    fb=lb**2/(2*np.pi)
                    
                    plt.errorbar(lb,mean*fb,std*fb,fmt='o')
                    plt.plot(lth,ps_th[spec]*lth**2/(2*np.pi))
                    plt.plot(lb,bin_theory[spec]*fb)
                    plt.show()

