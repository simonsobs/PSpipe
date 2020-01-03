"""
This script is used for producing the figure of the paper displaying the sensitivity to cosmological parameters of the cross correlation coeff
To run it: python cosmo_variation.py
It uses pycamb and compute the correlation coefficient for different values of the LCDM cosmological parameters
"""
import numpy as np, pylab as plt, matplotlib as mpl
import camb
from camb import model, initialpower
from pspy import pspy_utils

figure_dir='figures'
pspy_utils.create_directory(figure_dir)

pars = camb.CAMBparams()

#We set the cosmology to planck fiducial cosmology
p={}
p['H0']=67.36
p['ombh2']=0.02237
p['omch2']=0.120
p['tau']=0.0544
p['As']=2.1e-9
p['ns']=0.9649

#if use_H0==True, this will produce Figure 1 of the paper
# otherwise we will vary parameters keeping the sound horizon constant
use_H0=True

lmax=3500
nvalue=20 # number of parameters value for which the spectra are computed
max_range=0.2 #The range of variation of the parameter values (0.2=20%)

c = np.arange(1, nvalue + 1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

if use_H0==True:
    paramlist=['H0','ombh2','omch2','tau','As','ns']
    namelist=[r'$H_0$',r'$\Omega_{b}h^{2}$',r'$\Omega_{c}h^{2}$',r'$\tau$',r'$A_s$',r'$n_s$']
else:
    cosmomc_theta=0.0104092
    paramlist=['ombh2','omch2','tau','As','ns','cosmomc_theta']
    namelist=[r'$\Omega_{b}h^{2}$',r'$\Omega_{c}h^{2}$',r'$\tau$',r'$A_s$',r'$n_s$',r'$\theta_{MC}$']


fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(12,7))

count=0
for param,name,ax in zip(paramlist,namelist,axes.flat):
    
    if param=='cosmomc_theta':
        ax.set_axis_off()
        count+=1; continue
    
    range=np.linspace(p[param]-p[param]*max_range,p[param]+p[param]*max_range,nvalue)
    
    ax.set_title('%s'%name,fontsize=13)
    for cc,pvalue in enumerate(range):
        print (param,pvalue)
        p[param]=pvalue
        if use_H0==True:
            pars.set_cosmology(H0=p['H0'], ombh2=p['ombh2'], omch2=p['omch2'], mnu=0.06, omk=0, tau=p['tau'])
        else:
            pars.set_cosmology(cosmomc_theta=cosmomc_theta, ombh2=p['ombh2'], omch2=p['omch2'], mnu=0.06, omk=0, tau=p['tau'])
        
        pars.InitPower.set_params(As=p['As'], ns=p['ns'], r=0)
        pars.set_for_lmax(lmax, lens_potential_accuracy=0);
        results = camb.get_results(pars)
        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
        lensedCL=powers['lensed_scalar']
  
        TT,EE,TE=lensedCL[2:,0],lensedCL[2:,1],lensedCL[2:,3]
        ls = np.arange(2,TT.shape[0]+2)

        ax.plot(ls,TE/np.sqrt(TT*EE),color=cmap.to_rgba(cc + 1))
        if count==0:
            ax.set_ylabel(r'$R^{TE}_{\ell}$',fontsize=13)
        if count >2:
            ax.set_xlabel(r'$\ell$',fontsize=13)
    count+=1

cb=fig.colorbar(cmap, ax=axes.ravel().tolist(), ticks=[1, nvalue],shrink=0.6)
cb.ax.set_yticklabels(['-20%','+20%'],fontsize=13)
if use_H0==True:
    fig.savefig('%s/cosmo_dependency.pdf'%figure_dir,bbox_inches = 'tight')
else:
    fig.savefig('%s/fixed_theta_cosmo_dependency.pdf'%figure_dir,bbox_inches = 'tight')

plt.clf()
plt.close()
