from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
import  numpy as np, pylab as plt, healpy as hp
import os,sys
import so_noise_calculator_public_20180822 as noise_calc
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type=d['type']
freqs=d['freqs']
iStart=d['iStart']
iStop=d['iStop']
nSplits=d['nSplits']
lmax=d['lmax']
type=d['type']
clfile=d['clfile']
lcut=d['lcut']
hdf5=d['hdf5']

specDir='spectra'

if hdf5:
    spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'r')

mcm_dir='mcm'
plot_dir='plot'
pspy_utils.create_directory(plot_dir)

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

lth,Dlth=pspy_utils.ps_lensed_theory_to_dict(clfile,output_type=type,lmax=lmax,lstart=2)

theory={}
bin_theory={}
for fid1,f1 in enumerate(freqs):
    for fid2,f2 in enumerate(freqs):
        if fid1>fid2: continue
            
        Nl_file_T='noise_ps/noise_T_%sx%s.dat'%(f1,f2)
        Nl_file_P='noise_ps/noise_P_%sx%s.dat'%(f1,f2)
        l,bl1=np.loadtxt('beam/beam_%s.dat'%f1,unpack=True)
        l,bl2=np.loadtxt('beam/beam_%s.dat'%f2,unpack=True)

        nlth=maps_to_params_utils.get_effective_noise(lmax,bl1,bl2,Nl_file_T,Nl_file_P,spectra,lcut=lcut)

        prefix= '%s/%sx%s'%(mcm_dir,f1,f2)
        mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

        for kind in ['cross','noise','auto']:

            ps_th={}
            for spec in spectra:
                if kind=='cross':
                    ps_th[spec]=Dlth[spec]
                elif kind=='noise':
                    ps_th[spec]=nlth[spec]*lth**2/(2*np.pi)
                elif kind=='auto':
                    ps_th[spec]=Dlth[spec]+nlth[spec]*lth**2/(2*np.pi)*nSplits
    
            theory[f1,f2,kind]=ps_th
            bin_theory[f1,f2,kind]=so_mcm.apply_Bbl(Bbl,ps_th,spectra=spectra)

spec_name={}
for kind in ['cross','noise','auto']:
    vec_list=[]
    for iii in range(iStart,iStop):
        vec=[]
        count=0
        for spec in spectra:
            for fid1,f1 in enumerate(freqs):
                for fid2,f2 in enumerate(freqs):
                    if fid1>fid2: continue
                    spec_name='%s_%sx%s_%s_%05d'%(type,f1,f2,kind,iii)
                    
                    if hdf5:
                        lb,Db=so_spectra.read_ps_hdf5(spectra_hdf5,spec_name,spectra=spectra)
                    else:
                        lb,Db=so_spectra.read_ps(specDir+'/%s.dat'%spec_name,spectra=spectra)

                    
                    n_bins=len(lb)
                    vec=np.append(vec,Db[spec])
        vec_list+=[vec]
   
    mean_vec=np.mean(vec_list,axis=0)
    cov=0
    for iii in range(iStart,iStop):
        cov+=np.outer(vec_list[iii],vec_list[iii])
    cov=cov/(iStop-iStart)-np.outer(mean_vec, mean_vec)

    id_spec=0
    for spec in spectra:
        for fid1,f1 in enumerate(freqs):
            for fid2,f2 in enumerate(freqs):
                if fid1>fid2: continue
                
                mean=mean_vec[id_spec*n_bins:(id_spec+1)*n_bins]
                std=np.sqrt(cov[id_spec*n_bins:(id_spec+1)*n_bins,id_spec*n_bins:(id_spec+1)*n_bins].diagonal())
                
                plt.figure(figsize=(8,7))
                
                if spec=='TT':
                    plt.semilogy()
                
                plt.plot(lth,theory[f1,f2,kind][spec],color='grey',alpha=0.4)
                plt.plot(lb,bin_theory[f1,f2,kind][spec])
                plt.errorbar(lb,mean,std,fmt='.',color='red')
                plt.title(r'$D^{%s,%sx%s}_{%s,\ell}$'%(spec,f1,f2,kind),fontsize=20)
                plt.xlabel(r'$\ell$',fontsize=20)
                plt.savefig('plot/spectra_%s_%sx%s_%s.png'%(spec,f1,f2,kind),bbox_inches='tight')
                plt.clf()
                plt.close()
                
                plt.errorbar(lb,mean-bin_theory[f1,f2,kind][spec],std,fmt='.',color='red')
                plt.title(r'$D^{%s,%sx%s}_{%s,\ell}-D^{%s,%sx%s,th}_{%s,\ell}$'%(spec,f1,f2,kind,spec,f1,f2,kind),fontsize=20)
                plt.xlabel(r'$\ell$',fontsize=20)
                plt.savefig('plot/diff_spectra_%s_%sx%s_%s.png'%(spec,f1,f2,kind),bbox_inches='tight')
                plt.clf()
                plt.close()
                
                std/=np.sqrt(iStop-iStart)

                plt.errorbar(lb,(mean-bin_theory[f1,f2,kind][spec])/std,color='red')
                plt.title(r'$(D^{%s,%sx%s}_{%s,\ell}-D^{%s,%sx%s,th}_{%s,\ell})/\sigma$'%(spec,f1,f2,kind,spec,f1,f2,kind),fontsize=20)
                plt.xlabel(r'$\ell$',fontsize=20)
                plt.savefig('plot/frac_spectra_%s_%sx%s_%s.png'%(spec,f1,f2,kind),bbox_inches='tight')
                plt.clf()
                plt.close()


                id_spec+=1
