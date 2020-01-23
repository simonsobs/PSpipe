"""
This script is used to extract and plot the beam of the planck data
Planck provide window function computed over the Plick un-masked sky
The window function contains term such as TT_2_TT, TT_2_TE etc
Here we only extract sqrt(TT_2_TT) and sqrt(EE_2_EE) and call it beam_T and beam_pol we do it for hm1 and hm2 and all frequencies
We also 'extrapolate' the window function to higher ell (leaving it to a constant value bl[lmax]), this won't be used in the rest of the analysis.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from pspy import so_dict,so_map,pspy_utils
import sys
import astropy.io.fits as fits
from matplotlib.pyplot import cm

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

freqs=d['freqs']
splits=d['splits']
data_dir= d['data_dir']

mylmax=6000

plot_dir='plot'
pspy_utils.create_directory(plot_dir)

spectra=['TT','EE','BB','TE']
leakage_term={}
for spec in spectra:
    leakage_term[spec]=['%s_2_TT'%spec, '%s_2_EE'%spec,'%s_2_BB'%spec,'%s_2_TE'%spec, '%s_2_TB'%spec,'%s_2_EB'%spec,'%s_2_ET'%spec,'%s_2_BT'%spec,'%s_2_BE'%spec]

n=len(freqs)*len(splits)+1
color=iter(cm.rainbow(np.linspace(0,1,n)))

plt.figure(figsize=(12,12))
for f in freqs:
    for hm in splits:
        
        Wl=fits.open('%s/beams/BeamWf_HFI_R3.01/Wl_R3.01_plikmask_%s%sx%s%s.fits'%(data_dir,f,hm,f,hm))
        Wl_dict={}
        num=1
        for spec in spectra:
            for leak in leakage_term[spec]:
                Wl_dict[leak]=Wl[num].data[leak]
            num+=1
        
        lmax=len(Wl_dict['TT_2_TT'][0])

        bl_T=np.zeros(mylmax)
        bl_pol=np.zeros(mylmax)

        bl_T[:lmax]= np.sqrt(Wl_dict['TT_2_TT'][0])
        bl_pol[:lmax]= np.sqrt(Wl_dict['EE_2_EE'][0])

        bl_T[lmax:]=bl_T[lmax-1]
        bl_pol[lmax:]=bl_pol[lmax-1]

        l=np.arange(mylmax)
        
        np.savetxt('%s/beams/beam_T_%s_%s.dat'%(data_dir,f,hm), np.transpose([l,bl_T]))
        np.savetxt('%s/beams/beam_pol_%s_%s.dat'%(data_dir,f,hm), np.transpose([l,bl_pol]))
        
        c=next(color)
        plt.errorbar(l,bl_T,label='%s %s T'%(f,hm),color=c)
        plt.errorbar(l,bl_pol,label='%s %s pol'%(f,hm),color=c,fmt='--')

plt.legend()
plt.savefig('%s/beams.png'%plot_dir,bbox_inches='tight')
plt.clf()
plt.close()
