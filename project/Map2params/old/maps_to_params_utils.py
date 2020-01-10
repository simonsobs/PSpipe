#!/usr/bin/env python

import  numpy as np, healpy as hp
import os,sys
from pixell import curvedsky

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def get_noise_matrix_spin0and2(noise_dir,exp,freqs,lmax,nSplits,lcut=0):
    
    Nfreq=len(freqs)
    Nl_array_T=np.zeros((Nfreq,Nfreq,lmax))
    Nl_array_P=np.zeros((Nfreq,Nfreq,lmax))
    
    for c1,f1 in enumerate(freqs):
        for c2,f2 in enumerate(freqs):
            if c1>c2 : continue
            l,Nl_T=np.loadtxt('%s/noise_T_%s_%sx%s_%s.dat'%(noise_dir,exp,f1,exp,f2),unpack=True)
            l,Nl_P=np.loadtxt('%s/noise_P_%s_%sx%s_%s.dat'%(noise_dir,exp,f1,exp,f2),unpack=True)
            for i in range(lcut,lmax):
                Nl_array_T[c1,c2,i]=Nl_T[i]*nSplits
                Nl_array_P[c1,c2,i]=Nl_P[i]*nSplits
    for i in range(lmax):
        Nl_array_T[:,:,i]=symmetrize(Nl_array_T[:,:,i])
        Nl_array_P[:,:,i]=symmetrize(Nl_array_P[:,:,i])

    return(l,Nl_array_T,Nl_array_P)

def get_foreground_matrix(foreground_dir,extragal_foregrounds,allfreqs,lmax):
    
    Nfreq=len(allfreqs)
    Fl_array_T=np.zeros((Nfreq,Nfreq,lmax))
    for c1,f1 in enumerate(allfreqs):
        for c2,f2 in enumerate(allfreqs):
            if c1>c2 : continue
            Fl_all=0
            for foreground in extragal_foregrounds:
                l,Fl=np.loadtxt('%s/tt_%s_%sx%s.dat'%(foreground_dir,foreground,f1,f2),unpack=True)
                Fl_all+= Fl*2*np.pi/(l*(l+1))
            for i in range(2,lmax):
                Fl_array_T[c1,c2,i]=Fl_all[i-2]
    for i in range(lmax):
        Fl_array_T[:,:,i]=symmetrize(Fl_array_T[:,:,i])

    return(l,Fl_array_T)


def convolved_alms(alms,bl,ncomp):
    alms_convolved=alms.copy()
    if ncomp==1:
        alms_convolved=hp.sphtfunc.almxfl(alms_convolved,bl)
    else:
        for i in range(ncomp):
            alms_convolved[i]=hp.sphtfunc.almxfl(alms_convolved[i],bl)
    return alms_convolved

def generate_noise_alms(Nl_array_T,Nl_array_P,lmax,nSplits,ncomp):
    nlms={}
    if ncomp==1:
        for k in range(nSplits):
            nlms[k]=curvedsky.rand_alm(Nl_array_T,lmax=lmax)
    else:
        for k in range(nSplits):
            nlms['T',k]=curvedsky.rand_alm(Nl_array_T,lmax=lmax)
            nlms['E',k]=curvedsky.rand_alm(Nl_array_P,lmax=lmax)
            nlms['B',k]=curvedsky.rand_alm(Nl_array_P,lmax=lmax)
    
    return nlms

def remove_mean(map,window,ncomp):
    if ncomp==1:
        map.data-=np.mean(map.data*window.data)
    else:
        for i in range(ncomp):
            map.data[0]-=np.mean(map.data[0]*window[0].data)
            map.data[1]-=np.mean(map.data[1]*window[1].data)
            map.data[2]-=np.mean(map.data[2]*window[1].data)
    return map


def get_effective_noise(lmax,bl1,bl2,Nl_file_T,Nl_file_P,spectra,lcut=0):
    bl1=bl1[lcut:lmax]
    bl2=bl2[lcut:lmax]
    if spectra==None:
        l,noise_ps=np.loadtxt(Nl_file_T,unpack=True)
        noise_ps=np.zeros(lmax)
        noise_ps[lcut:lmax]/=(bl1*bl2)
    else:
        noise_ps={}
        for spec in spectra:
            noise_ps[spec]=np.zeros(lmax)
        l,noise_ps_T=np.loadtxt(Nl_file_T,unpack=True)
        l,noise_ps_P=np.loadtxt(Nl_file_P,unpack=True)
        noise_ps['TT'][lcut:lmax]=noise_ps_T[lcut:lmax]/(bl1*bl2)
        noise_ps['EE'][lcut:lmax]=noise_ps_P[lcut:lmax]/(bl1*bl2)
        noise_ps['BB'][lcut:lmax]=noise_ps_P[lcut:lmax]/(bl1*bl2)
    
    return noise_ps
