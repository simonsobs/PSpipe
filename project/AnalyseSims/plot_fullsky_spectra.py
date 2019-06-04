import matplotlib
matplotlib.use('Agg')
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

plot_dir='plot'
spec_dir='spectra'

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(spec_dir)

experiment=d['experiment']
content=d['content']
lmax=d['lmax']
spectra=['TT', 'EE', 'BB', 'TE', 'EB', 'TB']

cls={}
for exp in experiment:
    freqs=d['freq_%s'%exp]
    for cont in content:
        maps_list= d['%s_maps'%cont]
        for f in freqs:
            cl_array=np.loadtxt('%s/full_sky_cl_%s_%s_%s.dat'%(spec_dir,cont,exp,f))
            for count,spec in enumerate(spectra):
                cls[exp,cont,f,spec]=cl_array[:,count]

l=np.arange(lmax+1)
fac=l*(l+1)/(2*np.pi)

for spec in spectra:
    for cont in content:
        for exp in experiment:
            for f in freqs:
                if cont != 'noise':
                    l,bl= np.loadtxt(d['beam_%s'%f],unpack=True)
                    l,bl=l[:lmax+1],bl[:lmax+1]
                else:
                    bl=l*0+1
                plt.plot(l,cls[exp,cont,f,spec]*fac/bl**2,label='%s %s'%(exp,f))
        plt.legend()
        plt.savefig('%s/spec_%s_%s.png'%(plot_dir,cont,spec))
        plt.clf()
        plt.close()
