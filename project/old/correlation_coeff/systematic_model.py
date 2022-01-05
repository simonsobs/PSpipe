"""
This script generates fake beams and transfer function.
to run it:
python systematic_model.py
"""

import numpy as np, pylab as plt, matplotlib as mpl
from pspy import pspy_utils

syst_dir='systematics'
pspy_utils.create_directory(syst_dir)

FWHM= 7.30
FWHM_syst=0.95*FWHM

beam_FWHM_rad = np.deg2rad(FWHM_syst)/60
beam = beam_FWHM_rad/np.sqrt(8*np.log(2))
l=np.arange(2,5000)
bl=np.exp(-l*(l+1)*beam**2/2.)
np.savetxt('%s/beam.dat'%(syst_dir), np.transpose([l,bl]))

lmax_array=[400,200]
min_TF_array=[0.7,0.8]

cal=1.
pol_eff=1.

cal_array=[cal,cal*pol_eff]
name_array=['T','pol']

for name,lmax,min_TF,cal in zip(name_array,lmax_array,min_TF_array,cal_array):
    id=np.where(l<lmax)
    TF=l*0+1.
    TF[id]*=min_TF+(1-min_TF)*(np.cos((lmax - l[id])/(lmax-2)*np.pi/2.))**2
    TF*=cal
    np.savetxt('%s/transferfunction_%s.dat'%(syst_dir,name), np.transpose([l,TF]))


