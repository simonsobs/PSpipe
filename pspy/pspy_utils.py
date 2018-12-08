"""
@brief: utils for pspy.
"""
from __future__ import absolute_import, print_function
import healpy as hp, pylab as plt, numpy as np

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



