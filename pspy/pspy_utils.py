"""
@brief: utils for pspy.
"""
from __future__ import absolute_import, print_function
import healpy as hp, pylab as plt, numpy as np
import os

def ps_lensed_theory_to_dict(filename,output_type,lmax=None,lstart=2):
    """
    @brief read a lensed power spectrum from CAMB and return a dictionnary
    @param filename: the name of the CAMB lensed power spectrum
    @param lmax: the maximum multipole
    @param output_type: 'Cl' or 'Dl'
    @param lstart: choose the 0 entry of the ps spectrum, default is l=2, can be 0
    @return ps: a dictionnary file with the power spectra
    """
    fields=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    ps={}
    l,ps['TT'],ps['EE'],ps['BB'],ps['TE']=np.loadtxt(filename,unpack=True)
    ps['ET']=ps['TE']
    ps['TB'],ps['BT'],ps['EB'],ps['BE']=np.zeros((4,len(l)))
    
    if lmax is not None:
        l=l[:lmax]
    scale=l*(l+1)/(2*np.pi)
    for f in fields:
        if lmax is not None:
            ps[f]=ps[f][:lmax]
        if output_type=='Cl':
            ps[f]/=scale
        if lstart==0:
            ps[f]=np.append( np.array([0,0]),ps[f])
    if lstart==0:
        l=np.append( np.array([0,1]),l)
    return l,ps

def get_nlth_dict(rms_uKarcmin_T,type,lmax,spectra=None,rms_uKarcmin_pol=None,beamfile=None):
    """
    @brief return the effective noise power spectrum Nl/bl^2 given a beam file and a noise rms
    @param rms_uKarcmin_T: the temperature noise rms in uK.arcmin
    @param type: the type of binning, either bin Cl or bin Dl
    @param lmax: the maximum multipole to consider
    @param spectra: needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    @param (optional) rms_uKarcmin_pol: the temperature noise rms in uK.arcmin
    @param (optional) beamfile: the location of the beam transfer function (assuming it's given as a two column file l,bl)
    @return nl_th: a dictionnary file with effective noise power spectra
    """
    if beamfile is not None:
        l,bl=np.loadtxt(beamfile,unpack=True)
    else:
        bl=np.ones(lmax+2)

    lth=np.arange(2,lmax+2)
    nl_th={}
    if spectra is None:
        nl_th['TT']=np.ones(lmax)*(rms_uKarcmin_T*np.pi/(60*180))**2/bl[2:lmax+2]**2
        if type=='Dl':
            nl_th['TT']*=lth*(lth+1)/(2*np.pi)
        return nl_th
    else:
        if rms_uKarcmin_pol is None:
            rms_uKarcmin_pol=rms_uKarcmin_T*np.sqrt(2)
        for spec in spectra:
            nl_th[spec]=np.zeros(lmax)
        nl_th['TT']=np.ones(lmax)*(rms_uKarcmin_T*np.pi/(60*180))**2/bl[:lmax]**2
        nl_th['EE']=np.ones(lmax)*(rms_uKarcmin_pol*np.pi/(60*180))**2/bl[:lmax]**2
        nl_th['BB']=np.ones(lmax)*(rms_uKarcmin_pol*np.pi/(60*180))**2/bl[:lmax]**2
        if type=='Dl':
            for spec in spectra:
                nl_th[spec]*=lth*(lth+1)/(2*np.pi)
    return(nl_th)


def read_binning_file(file,lmax):
    """
    @brief read a binningFile and truncate it to lmax, if bin_low lower than 2, set it to 2.
    @param binningfile with format: bin_low,bin_high,bin_center
    @return bin_lo,bin_hi,bin_center,bin_size
    """
    bin_lo,bin_hi,bin_c = plt.loadtxt(file,unpack=True)
    id = np.where(bin_hi <lmax)
    bin_lo,bin_hi,bin_c=bin_lo[id],bin_hi[id],bin_c[id]
    if bin_lo[0]<2:
        bin_lo[0]=2
    bin_hi=bin_hi.astype(np.int)
    bin_lo=bin_lo.astype(np.int)
    bin_size=bin_hi-bin_lo+1
    return (bin_lo,bin_hi,bin_c,bin_size)

def create_directory(name):
    """
    @brief create a directory
    """
    try:
        os.makedirs(name)
    except:
        pass

def naive_binning(l,fl,binning_file,lmax):
    bin_lo,bin_hi,bin_c,bin_size= read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    fl_bin=np.zeros(len(bin_c))
    for ibin in range(n_bins):
        loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
        fl_bin[ibin] = (fl[loc]).mean()
    return bin_c,fl_bin

