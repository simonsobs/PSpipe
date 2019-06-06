from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
from matplotlib.pyplot import cm
import  numpy as np, pylab as plt, healpy as hp
import os,sys
import so_noise_calculator_public_20180822 as noise_calc
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type=d['type']
experiment=d['experiment']
iStart=d['iStart']
iStop=d['iStop']
lmax=d['lmax']
type=d['type']
clfile=d['clfile']
lcut=d['lcut']
hdf5=d['hdf5']
multistep_path=d['multistep_path']

foreground_dir=d['foreground_dir']
extragal_foregrounds=d['extragal_foregrounds']

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

ns={}
for exp in experiment:
    ns[exp]=2

for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue
    
                l,bl1=np.loadtxt('beam/beam_%s_%s.dat'%(exp1,f1),unpack=True)
                l,bl2=np.loadtxt('beam/beam_%s_%s.dat'%(exp2,f2),unpack=True)

                if exp1==exp2:
                    Nl_file_T='noise_ps/noise_T_%s_%sx%s_%s.dat'%(exp1,f1,exp2,f2)
                    Nl_file_P='noise_ps/noise_P_%s_%sx%s_%s.dat'%(exp1,f1,exp2,f2)
                    nlth=maps_to_params_utils.get_effective_noise(lmax,bl1,bl2,Nl_file_T,Nl_file_P,spectra,lcut=lcut)
                else:
                    nlth={}
                    for spec in spectra:
                        nlth[spec]=np.zeros(lmax)

                prefix= '%s/%s_%sx%s_%s'%(mcm_dir,exp1,f1,exp2,f2)
                mbb_inv,Bbl=so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

                for kind in ['cross','noise','auto']:
                    ps_th={}
                    for spec in spectra:
                        
                        ps=Dlth[spec].copy()
                        
                        if spec=='TT':
                            flth_all=0
                            for foreground in extragal_foregrounds:
                                l,flth=np.loadtxt('%s/%s_%sx%s.dat'%(foreground_dir,foreground,f1,f2),unpack=True)
                                flth_all+=flth[:lmax]
                            ps=Dlth[spec]+flth_all
                    
                        if kind=='cross':
                            ps_th[spec]=ps
                        elif kind=='noise':
                            ps_th[spec]=nlth[spec]*lth**2/(2*np.pi)
                        elif kind=='auto':
                            ps_th[spec]=ps+nlth[spec]*lth**2/(2*np.pi)*ns[exp1]
    
                    theory[exp1,f1,exp2,f2,kind]=ps_th
                    bin_theory[exp1,f1,exp2,f2,kind]=so_mcm.apply_Bbl(Bbl,ps_th,spectra=spectra)

spec_name={}
mean_dict={}
std_dict={}
n_spec={}
for kind in ['cross','noise','auto']:
    vec_list=[]
    for iii in range(iStart,iStop):
        vec=[]
        count=0
        for spec in spectra:
            for id_exp1,exp1 in enumerate(experiment):
                freqs1=d['freq_%s'%exp1]
                for id_f1,f1 in enumerate(freqs1):
                    for id_exp2,exp2 in enumerate(experiment):
                        freqs2=d['freq_%s'%exp2]
                        for id_f2,f2 in enumerate(freqs2):
                            if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                            if  (id_exp1>id_exp2) : continue
                            if (exp1!=exp2) & (kind=='noise'): continue
                            if (exp1!=exp2) & (kind=='auto'): continue

                            spec_name='%s_%s_%sx%s_%s_%s_%05d'%(type,exp1,f1,exp2,f2,kind,iii)
                    
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
        
        n_spec[kind]=0

        for id_exp1,exp1 in enumerate(experiment):
            freqs1=d['freq_%s'%exp1]
            for id_f1,f1 in enumerate(freqs1):
                for id_exp2,exp2 in enumerate(experiment):
                    freqs2=d['freq_%s'%exp2]
                    for id_f2,f2 in enumerate(freqs2):
                        if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                        if  (id_exp1>id_exp2) : continue
                        if (exp1!=exp2) & (kind=='noise'): continue
                        if (exp1!=exp2) & (kind=='auto'): continue

                
                        mean_dict[kind,spec,exp1,f1,exp2,f2]=mean_vec[id_spec*n_bins:(id_spec+1)*n_bins]
                        std_dict[kind,spec,exp1,f1,exp2,f2]=np.sqrt(cov[id_spec*n_bins:(id_spec+1)*n_bins,id_spec*n_bins:(id_spec+1)*n_bins].diagonal())
                
                        mean=mean_dict[kind,spec,exp1,f1,exp2,f2]
                        std=std_dict[kind,spec,exp1,f1,exp2,f2]

                        plt.figure(figsize=(8,7))
                
                        if spec=='TT':
                            plt.semilogy()
                
                        plt.plot(lth,theory[exp1,f1,exp2,f2,kind][spec],color='grey',alpha=0.4)
                        plt.plot(lb,bin_theory[exp1,f1,exp2,f2,kind][spec])
                        plt.errorbar(lb,mean,std,fmt='.',color='red')
                        plt.title(r'$D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}$'%(spec,exp1,f1,exp2,f2,kind),fontsize=20)
                        plt.xlabel(r'$\ell$',fontsize=20)
                        plt.savefig('plot/spectra_%s_%s_%sx%s_%s_%s.png'%(spec,exp1,f1,exp2,f2,kind),bbox_inches='tight')
                        plt.clf()
                        plt.close()
                
                        plt.errorbar(lb,mean-bin_theory[exp1,f1,exp2,f2,kind][spec],std,fmt='.',color='red')
                        plt.title(r'$D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}-D^{%s,%s_{%s}x%s_{%s},th}_{%s,\ell}$'%(spec,exp1,f1,exp2,f2,kind,spec,exp1,f1,exp2,f2,kind),fontsize=20)
                        plt.xlabel(r'$\ell$',fontsize=20)
                        plt.savefig('plot/diff_spectra_%s_%s_%sx%s_%s_%s.png'%(spec,exp1,f1,exp2,f2,kind),bbox_inches='tight')
                        plt.clf()
                        plt.close()
                
                        std/=np.sqrt(iStop-iStart)

                        plt.errorbar(lb,(mean-bin_theory[exp1,f1,exp2,f2,kind][spec])/std,color='red')
                        plt.title(r'$(D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}-D^{%s,%s_{%s}x%s_{%s},th}_{%s,\ell})/\sigma$'%(spec,exp1,f1,exp2,f2,kind,spec,exp1,f1,exp2,f2,kind),fontsize=20)
                        plt.xlabel(r'$\ell$',fontsize=20)
                        plt.savefig('plot/frac_spectra_%s_%s_%sx%s_%s_%s.png'%(spec,exp1,f1,exp2,f2,kind),bbox_inches='tight')
                        plt.clf()
                        plt.close()

                        id_spec+=1
                        n_spec[kind]+=1


plt.figure(figsize=(12,12))
color=iter(cm.rainbow(np.linspace(0,1,n_spec['cross']+1)))


for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue

                c=next(color)
                plt.errorbar(lb,mean_dict['cross','TT',exp1,f1,exp2,f2],std_dict['cross','TT',exp1,f1,exp2,f2],fmt='.',color=c,label='%s%s x %s%s'%(exp1,f1,exp2,f2))
                plt.plot(lth,theory[exp1,f1,exp2,f2,'cross']['TT'],color=c,alpha=0.4)

plt.legend()
plt.xlabel(r'$\ell$',fontsize=20)
plt.savefig('plot/all_spectra_TT_%s_%sx%s_%s_cross.png'%(exp1,f1,exp2,f2),bbox_inches='tight')
plt.clf()
plt.close()



os.system('cp %s/multistep2.js %s/multistep2.js'%(multistep_path,plot_dir))
fileName='%s/SO_spectra.html'%plot_dir
g = open(fileName,mode="w")
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> SO spectra </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
g.write('<script> add_step("all",  ["j","k"]) </script> \n')
g.write('<script> add_step("type",  ["a","z"]) </script> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub> \n')

for kind in ['cross','noise','auto']:
    g.write('<div class=all>\n')
    for spec in spectra:
        for id_exp1,exp1 in enumerate(experiment):
            freqs1=d['freq_%s'%exp1]
            for id_f1,f1 in enumerate(freqs1):
                for id_exp2,exp2 in enumerate(experiment):
                    freqs2=d['freq_%s'%exp2]
                    for id_f2,f2 in enumerate(freqs2):
                        if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                        if  (id_exp1>id_exp2) : continue
                        if (exp1!=exp2) & (kind=='noise'): continue
                        if (exp1!=exp2) & (kind=='auto'): continue


                        str='spectra_%s_%s_%sx%s_%s_%s.png'%(spec,exp1,f1,exp2,f2,kind)
                        g.write('<div class=type>\n')
                        g.write('<img src="'+str+'" width="50%" /> \n')
                        g.write('<img src="'+'diff_'+str+'" width="50%" /> \n')
                        g.write('<img src="'+'frac_'+str+'" width="50%" /> \n')
                        g.write('</div>\n')

    g.write('</div>\n')
g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()


