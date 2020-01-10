import matplotlib
matplotlib.use('Agg')
from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
from matplotlib.pyplot import cm
import  numpy as np, pylab as plt, healpy as hp
import os,sys
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

include_fg=d['include_fg']
fg_dir=d['fg_dir']
fg_components=d['fg_components']


specDir='spectra'

if hdf5:
    spectra_hdf5 = h5py.File('%s.hdf5'%(specDir), 'r')

mcm_dir='mcm'
plot_dir='plot'
mc_dir='monteCarlo'

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
                            if include_fg:
                                flth_all=0
                                for foreground in fg_components:
                                    l,flth=np.loadtxt('%s/tt_%s_%sx%s.dat'%(fg_dir,foreground,f1,f2),unpack=True)
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

mean_dict={}
std_dict={}
n_spec={}
for kind in ['cross','noise','auto']:
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

                
                
                        lb,mean_dict[kind,spec,exp1,f1,exp2,f2],std_dict[kind,spec,exp1,f1,exp2,f2]=np.loadtxt('%s/spectra_%s_%s_%sx%s_%s_%s.dat'%(mc_dir,spec,exp1,f1,exp2,f2,kind),unpack=True)
                        
                        mean=mean_dict[kind,spec,exp1,f1,exp2,f2]
                        std=std_dict[kind,spec,exp1,f1,exp2,f2].copy()


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



lcut={}
lmax_spec={}
lmax_spec['Planck','100']=1800
lmax_spec['Planck','143']=2000
lmax_spec['Planck','217']=2300
lmax_spec['LAT','93']=lmax
lmax_spec['LAT','145']=lmax
lmax_spec['LAT','225']=lmax
for id_exp1,exp1 in enumerate(experiment):
    freqs1=d['freq_%s'%exp1]
    for id_f1,f1 in enumerate(freqs1):
        for id_exp2,exp2 in enumerate(experiment):
            freqs2=d['freq_%s'%exp2]
            for id_f2,f2 in enumerate(freqs2):
                if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                if  (id_exp1>id_exp2) : continue
                
                
                if f1=='100' or f2=='100':
                    lcut[exp1,f1,exp2,f2]=1800
                else:
                    lcut[exp1,f1,exp2,f2]=np.int((lmax_spec[exp1,f1]+ lmax_spec[exp2,f2])/2)



for fig in ['log','linear']:
    for spec in ['TT','TE','EE']:

        plt.figure(figsize=(12,12))
        color=iter(cm.rainbow(np.linspace(0,1,n_spec['cross']+1)))
    
        if fig=='log':
            plt.semilogy()

        exp_name=''
        for id_exp1,exp1 in enumerate(experiment):
            freqs1=d['freq_%s'%exp1]
            for id_f1,f1 in enumerate(freqs1):
                for id_exp2,exp2 in enumerate(experiment):
                    freqs2=d['freq_%s'%exp2]
                    for id_f2,f2 in enumerate(freqs2):
                        if  (id_exp1==id_exp2) & (id_f1>id_f2) : continue
                        if  (id_exp1>id_exp2) : continue
                
                        lmax= lcut[exp1,f1,exp2,f2]
                
                        id=np.where(lb<lmax)
                
                        mylb=lb[id]
                        mean_dict['cross',spec,exp1,f1,exp2,f2]=mean_dict['cross',spec,exp1,f1,exp2,f2][id]
                        std_dict['cross',spec,exp1,f1,exp2,f2]=std_dict['cross',spec,exp1,f1,exp2,f2][id]

                        c=next(color)
                    
                        if (fig=='linear') and (spec=='TT'):
                            plt.errorbar(mylb,mean_dict['cross',spec,exp1,f1,exp2,f2]*mylb**2,std_dict['cross',spec,exp1,f1,exp2,f2]*mylb**2,fmt='.',color=c,label='%s%s x %s%s'%(exp1,f1,exp2,f2),alpha=0.6)
                            plt.plot(lth[:lmax],theory[exp1,f1,exp2,f2,'cross'][spec][:lmax]*lth[:lmax]**2,color=c,alpha=0.4)
                        else:
                            plt.errorbar(mylb,mean_dict['cross',spec,exp1,f1,exp2,f2],std_dict['cross',spec,exp1,f1,exp2,f2],fmt='.',color=c,label='%s%s x %s%s'%(exp1,f1,exp2,f2),alpha=0.6)
                            plt.plot(lth[:lmax],theory[exp1,f1,exp2,f2,'cross'][spec][:lmax],color=c,alpha=0.4)

            exp_name+='%s_'%exp1

        if (fig=='log') and (spec=='TT'):
            plt.ylim(10,10**4)
        if (fig=='linear') and (spec=='TT'):
            plt.ylim(0,2*10**9)

        if fig=='log':
            plt.legend(fontsize=14,bbox_to_anchor=(1.4, 1.1))
        else:
            plt.legend(fontsize=14,bbox_to_anchor=(1.4, 1.))

        plt.xlabel(r'$\ell$',fontsize=20)

        if (fig=='linear') and (spec=='TT'):
            plt.ylabel(r'$\ell^{2} D^{%s}_\ell$'%spec,fontsize=20)
        else:
            plt.ylabel(r'$ D^{%s}_\ell$'%spec,fontsize=20)

        plt.savefig('plot/all_%s_spectra_%s_all_%scross.png'%(fig,spec,exp_name),bbox_inches='tight')
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


