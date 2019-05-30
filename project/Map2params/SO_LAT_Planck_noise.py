#!/usr/bin/env python

from pspy import pspy_utils, so_dict
import  numpy as np, pylab as plt
import os,sys
import so_noise_calculator_public_20180822 as noise_calc

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

freqs=d['freqs']
sensitivity_mode=d['sensitivity_mode'] #goal
f_sky_LAT= d['f_sky_LAT']
ell_min,ell_max=2,10000
delta_ell=1
pspy_utils.create_directory('noise_ps')

Nell_T={}
Nell_P={}

freqPairs_SO=['LAT27xLAT27','LAT39xLAT39','LAT93xLAT93','LAT145xLAT145','LAT225xLAT225','LAT280xLAT280','LAT27xLAT39','LAT93xLAT145','LAT225xLAT280']
ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels=noise_calc.Simons_Observatory_V3_LA_noise(sensitivity_mode,f_sky_LAT,ell_min,ell_max,delta_ell,N_LF=1.,N_MF=4.,N_UHF=2., apply_beam_correction=False, apply_kludge_correction=True)

for c1,f1 in enumerate(freqs):
    for c2,f2 in enumerate(freqs):
        if c1>c2 : continue
        freqPair='%sx%s'%(f1,f2)
        Nell_T[freqPair]=ell*0
        Nell_P[freqPair]=ell*0

for c,freqPair in enumerate(freqPairs_SO):
    Nell_T[freqPair]=N_ell_T_LA[c]
    Nell_P[freqPair]=N_ell_P_LA[c]

# Taken from Table 4: https://arxiv.org/pdf/1807.06205.pdf
sigma={}
sigma['Planck100xPlanck100']=77.4
sigma['Planck143xPlanck143']=33.0
sigma['Planck217xPlanck217']=46.80
sigma['Planck353xPlanck353']=153.6

sigmaP={}
sigmaP['Planck100xPlanck100']=117.6
sigmaP['Planck143xPlanck143']=70.2
sigmaP['Planck217xPlanck217']=105.0
sigmaP['Planck353xPlanck353']=438.6

freqPairs_Planck=['Planck100xPlanck100','Planck143xPlanck143', 'Planck217xPlanck217', 'Planck353xPlanck353']
for freqPair in freqPairs_Planck:
    sigma_rad = np.deg2rad(sigma[freqPair])/60
    Nell_T[freqPair]= ell*0+sigma_rad**2
    sigmaP_rad = np.deg2rad(sigmaP[freqPair])/60
    Nell_P[freqPair]= ell*0+sigmaP_rad**2

for c1,f1 in enumerate(freqs):
    for c2,f2 in enumerate(freqs):
        if c1>c2 : continue
        freqPair='%sx%s'%(f1,f2)
        np.savetxt('noise_ps/noise_T_%s.dat'%freqPair, np.transpose([ell,Nell_T[freqPair]]))
        np.savetxt('noise_ps/noise_P_%s.dat'%freqPair, np.transpose([ell,Nell_P[freqPair]]))

