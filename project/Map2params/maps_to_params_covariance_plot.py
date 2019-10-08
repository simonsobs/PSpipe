import matplotlib
matplotlib.use('Agg')
from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra,so_cov
import  numpy as np, pylab as plt, healpy as hp
import os,sys
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir='window'
mcm_dir='mcm'
cov_dir='covariance'
cov_plot_dir='plot_covariance'
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
multistep_path=d['multistep_path']
foreground_dir=d['foreground_dir']
extragal_foregrounds=d['extragal_foregrounds']


pspy_utils.create_directory(cov_dir)
pspy_utils.create_directory(cov_plot_dir)

if hdf5:
    spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'r')

ncomp=3
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0', 'spin2xspin2']

bin_lo,bin_hi,lb,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
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

                spec_name+=['%s_%sx%s_%s'%(exp1,f1,exp2,f2)]


lcut={}
lmax_spec={}
lmax_spec['Planck_100']=2000
lmax_spec['Planck_143']=2000
lmax_spec['Planck_217']=2000
lmax_spec['LAT_93']=2000
lmax_spec['LAT_145']=2000
lmax_spec['LAT_225']=2000


def cut_matrix(mat,cut_lmax,lb):
    nbins=int(mat.shape[0]/3)
    id=np.where(lb<cut_lmax)
    new_lb=lb[id]
    new_nbin=len(new_lb)
    new_mat=np.zeros((3*new_nbin,3*new_nbin))
    for i in range(3):
        for j in range(3):
            new_mat[i*new_nbin:(i+1)*new_nbin,j*new_nbin:(j+1)*new_nbin]=mat[i*nbins:i*nbins+new_nbin,j*nbins:j*nbins+new_nbin]
    return new_nbin,new_lb,new_mat


cov_all={}
analytic_cov_all={}

nspecs=0
nbins=0
for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        
        n1,n2=spec1.split('x')
        n3,n4=spec2.split('x')
        
        analytic_cov=np.load('%s/analytic_cov_%sx%s_%sx%s.npy'%(cov_dir,n1,n2,n3,n4) )
        cov=np.load('%s/mc_cov_%sx%s_%sx%s.npy'%(cov_dir,n1,n2,n3,n4))

        cut_lmax=np.amin([lmax_spec[n1],lmax_spec[n2],lmax_spec[n3],lmax_spec[n4]])
        
        new_nbin,new_lb,analytic_cov=cut_matrix(analytic_cov,cut_lmax,lb)
        new_nbin,new_lb,cov=cut_matrix(cov,cut_lmax,lb)
        
        
        for block in ['TTTT','EEEE','TETE','TTEE','TEEE','TETT','EETT','EETE','TTTE']:
            analytic_cov_all[spec1,spec2,block]=so_cov.selectblock(analytic_cov, ['TT','TE','EE'],new_nbin,block=block)
            cov_all[spec1,spec2,block]=so_cov.selectblock(cov, ['TT','TE','EE'],new_nbin,block=block)


for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        

        plt.figure(figsize=(15,15))
        plt.suptitle('%s %s (press c/v to switch between covariance matrix elements)'%(spec1,spec2),fontsize=30)
        count=1
        for bl in ['TTTT','EEEE','TETE','TTEE','TEEE','TETT']:
            plt.subplot(2,3,count)
            cov_select=cov_all[spec1,spec2,bl]
            analytic_cov_select=analytic_cov_all[spec1,spec2,bl]

            var = cov_select.diagonal()
            analytic_var = analytic_cov_select.diagonal()
            if count==1:
                plt.semilogy()
            plt.plot(new_lb[1:],var[1:],'o',label='MC %sx%s'%(bl[:2],bl[2:4]))
            plt.plot(new_lb[1:],analytic_var[1:],label='Analytic %sx%s'%(bl[:2],bl[2:4]))
            if count==1 or count==4:
                plt.ylabel(r'$Cov_{i,i,\ell}$',fontsize=22)
            if count >3:
                plt.xlabel(r'$\ell$',fontsize=22)
            plt.legend()
            count+=1
        plt.savefig('%s/cov_element_comparison_%s_%s.png'%(cov_plot_dir,spec1,spec2),bbox_inches='tight')
        plt.clf()
        plt.close()


        plt.figure(figsize=(13,9))
        count=1
        for bl in ['TTTT','EEEE','TETE','TTEE','TEEE','TETT','EETT','EETE','TTTE']:
            cov_select=cov_all[spec1,spec2,bl]
            analytic_cov_select=analytic_cov_all[spec1,spec2,bl]
            cov_select=np.log(np.abs(cov_select))
            analytic_cov_select=np.log(np.abs(analytic_cov_select))
            vmin,vmax=np.min(analytic_cov_select),np.max(analytic_cov_select)
            plt.subplot(3,6,count)
            plt.title('MC %s'%(bl),fontsize=6)
            plt.imshow(cov_select, vmin=vmin,vmax=vmax )
            count+=1
            plt.subplot(3,6,count)
            plt.title(' %s'%(bl),fontsize=6)
            plt.imshow(analytic_cov_select, vmin=vmin,vmax=vmax )
            count+=1
        
        plt.suptitle('%s %s'%(spec1,spec2))

        plt.savefig('%s/cov_element_comparison_%s_%s_matrix_log.png'%(cov_plot_dir,spec1,spec2),bbox_inches='tight')
        plt.clf()
        plt.close()





os.system('cp %s/multistep2.js %s/multistep2.js'%(multistep_path,cov_plot_dir))
fileName='%s/SO_covariance_pseudo_diagonal.html'%cov_plot_dir
g = open(fileName,mode="w")
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> SO covariance </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
g.write('<style> \n')
g.write('body { text-align: center; } \n')
g.write('img { width: 100%; max-width: 1200px; } \n')
g.write('</style> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub> \n')

for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        
        n1,n2=spec1.split('x')
        n3,n4=spec2.split('x')
                
        str='cov_element_comparison_%s_%s.png'%(spec1,spec2)
        g.write('<div class=sub>\n')
        g.write('<img src="'+str+'"  /> \n')
        g.write('</div>\n')

g.write('</body> \n')
g.write('</html> \n')
g.close()



os.system('cp %s/multistep2.js %s/multistep2.js'%(multistep_path,cov_plot_dir))
fileName='%s/SO_covariance.html'%cov_plot_dir
g = open(fileName,mode="w")
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> SO covariance </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
g.write('<style> \n')
g.write('body { text-align: center; } \n')
g.write('img { width: 100%; max-width: 1200px; } \n')
g.write('</style> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub> \n')

for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1>sid2: continue
        
        n1,n2=spec1.split('x')
        n3,n4=spec2.split('x')
        
        str='cov_element_comparison_%s_%s_matrix_log.png'%(spec1,spec2)
        g.write('<div class=sub>\n')
        g.write('<img src="'+str+'"  /> \n')
        g.write('</div>\n')

g.write('</body> \n')
g.write('</html> \n')
g.close()









