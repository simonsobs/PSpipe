'''
Some utility functions for processing the planck data
'''

import numpy as np,healpy as hp,pylab as plt
from pspy import so_dict, so_map,so_mcm,sph_tools,so_spectra,pspy_utils, so_map_preprocessing
import os,sys
from pixell import enmap
import time
from pixell import curvedsky


def process_planck_spectra(l,cl,binning_file,lmax,type,spectra=None,mcm_inv=None):
    bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    fac=(l*(l+1)/(2*np.pi))
    unbin_vec=[]
    mcm_inv=so_mcm.coupling_dict_to_array(mcm_inv)
    for f in spectra:
        unbin_vec=np.append(unbin_vec,cl[f][2:lmax])
    cl=so_spectra.vec2spec_dict(lmax-2,np.dot(mcm_inv,unbin_vec),spectra)
    l=np.arange(2,lmax)
    print (l.shape,cl['TT'].shape)
    vec=[]
    for f in spectra:
        binnedPower=np.zeros(len(bin_c))
        for ibin in range(n_bins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
            binnedPower[ibin] = (cl[f][loc]*fac[loc]).mean()/(fac[loc].mean())
        vec=np.append(vec,binnedPower)
    return l,cl,bin_c,so_spectra.vec2spec_dict(n_bins,vec,spectra)


def subtract_mono_di(map_in,mask_in, nside):
    #Taken from Zack script to remove monopole and dipole
    map_masked = hp.ma(map_in)
    map_masked.mask = (mask_in<1)
    mono, dipole = hp.pixelfunc.fit_dipole(map_masked)
    print(mono, dipole)
    m = map_in.copy()
    npix = hp.nside2npix(nside)
    bunchsize = npix // 24
    bad = hp.UNSEEN
    for ibunch in range(npix // bunchsize):
        ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
        ipix = ipix[(np.isfinite(m.flat[ipix]))]
        x, y, z = hp.pix2vec(nside, ipix, False)
        m.flat[ipix] -= dipole[0] * x
        m.flat[ipix] -= dipole[1] * y
        m.flat[ipix] -= dipole[2] * z
        m.flat[ipix] -= mono
    return m

def get_noise_matrix_spin0and2(noise_dir,exp,freqs,lmax,nSplits,lcut=0,use_noise_th=False):
    
    Nfreq=len(freqs)
    Nl_array_T=np.zeros((Nfreq,Nfreq,lmax))
    Nl_array_P=np.zeros((Nfreq,Nfreq,lmax))
    
    for c1,f1 in enumerate(freqs):
        for c2,f2 in enumerate(freqs):
            if c1 !=c2 : continue
            
            if use_noise_th==True:
                l,Nl_T=np.loadtxt('%s/noise_T_th_mean_%s_%sx%s_%s.dat'%(noise_dir,exp,f1,exp,f2),unpack=True)
                l,Nl_P=np.loadtxt('%s/noise_P_th_mean_%s_%sx%s_%s.dat'%(noise_dir,exp,f1,exp,f2),unpack=True)
            else:
                l,Nl_T=np.loadtxt('%s/noise_T_mean_%s_%sx%s_%s.dat'%(noise_dir,exp,f1,exp,f2),unpack=True)
                l,Nl_P=np.loadtxt('%s/noise_P_mean_%s_%sx%s_%s.dat'%(noise_dir,exp,f1,exp,f2),unpack=True)

            
            for i in range(lcut,lmax):
                Nl_array_T[c1,c2,i]=Nl_T[i]*nSplits
                Nl_array_P[c1,c2,i]=Nl_P[i]*nSplits
    for i in range(lmax):
        Nl_array_T[:,:,i]=symmetrize(Nl_array_T[:,:,i])
        Nl_array_P[:,:,i]=symmetrize(Nl_array_P[:,:,i])
    
    return(l,Nl_array_T,Nl_array_P)

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

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def binning(l,cl,lmax,binning_file=None,size=None):
    
    if binning_file is not None:
        bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    else:
        bin_lo=np.arange(2,lmax,size)
        bin_hi=bin_lo+size-1
        bin_c=(bin_lo+bin_hi)/2
    
    fac=(l*(l+1)/(2*np.pi))
    n_bins=len(bin_hi)
    binnedPower=np.zeros(len(bin_c))
    for ibin in range(n_bins):
        loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
        binnedPower[ibin] = (cl[loc]*fac[loc]).mean()/(fac[loc].mean())
    return bin_c,binnedPower
