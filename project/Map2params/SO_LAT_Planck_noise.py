#!/usr/bin/env python

from pspy import pspy_utils, so_dict
import  numpy as np, pylab as plt
import os,sys
import so_noise_calculator_public_20180822 as noise_calc

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiment=d['experiment']
sensitivity_mode=d['sensitivity_mode'] #goal
f_sky_LAT= d['f_sky_LAT']
ell_min,ell_max=2,10000
delta_ell=1
pspy_utils.create_directory('noise_ps')

Nell_T={}
Nell_P={}

freqPairs_SOLAT=['LAT_27xLAT_27','LAT_39xLAT_39','LAT_93xLAT_93','LAT_145xLAT_145','LAT_225xLAT_225','LAT_280xLAT_280','LAT_27xLAT_39','LAT_93xLAT_145','LAT_225xLAT_280']
ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels=noise_calc.Simons_Observatory_V3_LA_noise(sensitivity_mode,f_sky_LAT,ell_min,ell_max,delta_ell,N_LF=1.,N_MF=4.,N_UHF=2., apply_beam_correction=False, apply_kludge_correction=True)


freqs_LAT=d['freq_LAT']
for c1,f1 in enumerate(freqs_LAT):
    for c2,f2 in enumerate(freqs_LAT):
        if c1>c2 : continue
        freqPair='LAT_%sxLAT_%s'%(f1,f2)
        Nell_T[freqPair]=ell*0
        Nell_P[freqPair]=ell*0

for c,freqPair in enumerate(freqPairs_SOLAT):
    Nell_T[freqPair]=N_ell_T_LA[c]
    Nell_P[freqPair]=N_ell_P_LA[c]

# Taken from Table 4: https://arxiv.org/pdf/1807.06205.pdf
sigma={}
sigma['Planck_100xPlanck_100']=77.4
sigma['Planck_100xPlanck_143']=0.0
sigma['Planck_100xPlanck_217']=0.0
sigma['Planck_100xPlanck_353']=0.0
sigma['Planck_143xPlanck_143']=33.0
sigma['Planck_143xPlanck_217']=0.0
sigma['Planck_143xPlanck_353']=0.0
sigma['Planck_217xPlanck_217']=46.80
sigma['Planck_217xPlanck_353']=0.0
sigma['Planck_353xPlanck_353']=153.6

sigmaP={}
sigmaP['Planck_100xPlanck_100']=117.6
sigmaP['Planck_100xPlanck_143']=0.0
sigmaP['Planck_100xPlanck_217']=0.0
sigmaP['Planck_100xPlanck_353']=0.0
sigmaP['Planck_143xPlanck_143']=70.2
sigmaP['Planck_143xPlanck_217']=0.0
sigmaP['Planck_143xPlanck_353']=0.0
sigmaP['Planck_217xPlanck_217']=105.0
sigmaP['Planck_217xPlanck_353']=0.0
sigmaP['Planck_353xPlanck_353']=438.6



freqPairs_Planck=['Planck_100xPlanck_100','Planck_100xPlanck_143','Planck_100xPlanck_217','Planck_100xPlanck_353', 'Planck_143xPlanck_143','Planck_143xPlanck_217','Planck_143xPlanck_353', 'Planck_217xPlanck_217', 'Planck_217xPlanck_353','Planck_353xPlanck_353']
for freqPair in freqPairs_Planck:
    sigma_rad = np.deg2rad(sigma[freqPair])/60
    Nell_T[freqPair]= ell*0+sigma_rad**2
    sigmaP_rad = np.deg2rad(sigmaP[freqPair])/60
    Nell_P[freqPair]= ell*0+sigmaP_rad**2

for exp in experiment:
    freqs=d['freq_%s'%exp]
    for c1,f1 in enumerate(freqs):
        for c2,f2 in enumerate(freqs):
            if c1>c2 : continue
            freqPair='%s_%sx%s_%s'%(exp,f1,exp,f2)
            print (exp,freqPair)
            np.savetxt('noise_ps/noise_T_%s.dat'%(freqPair), np.transpose([ell,Nell_T[freqPair]]))
            np.savetxt('noise_ps/noise_P_%s.dat'%(freqPair), np.transpose([ell,Nell_P[freqPair]]))

