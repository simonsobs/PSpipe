"""
@brief: utils for pspy.
"""
import healpy as hp, pylab as plt, numpy as np

def Dl_lensed_theory_to_dict(filename,lmax,spin0=False):
    """
    @brief read a lensed power spectrum from CAMB and return a dictionnary
    @param Bbl: a binning matrix, if fields is not None will be a dictionnary, otherwise a (n_bins,lmax) matrix
    @param ps: a theoretical power spectrum: if fields is not None will be a dictionnary, otherwise a (lmax) vector
    @return Dbth: a theoretical binned power spectrum
    """
    fields=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    Dl={}
    l,Dl['TT'],Dl['EE'],Dl['BB'],Dl['TE']=np.loadtxt(filename,unpack=True)
    Dl['ET']=Dl['TE']
    Dl['TB'],Dl['BT'],Dl['EB'],Dl['BE']=np.zeros((4,len(l)))
    for f in fields:
        Dl[f]=Dl[f][:lmax]
    if spin0==True:
        return l,Dl['TT']
    else:
        return l,Dl

def read_binning_file(file,lmax):
    """
    @brief read a binningFile and truncate it to lmax, if bin_low lower than 2, set it to 2.
    @param binningfile with format: bin_low,bin_high,bin_center
    @return bin_lo,bin_hi,bin_center
    """
    bin_lo,bin_hi,bin_c = plt.loadtxt(file,unpack=True)
    id = np.where(bin_hi <lmax)
    bin_lo,bin_hi,bin_c=bin_lo[id],bin_hi[id],bin_c[id]
    if bin_lo[0]<2:
        bin_lo[0]=2
    bin_hi=bin_hi.astype(np.int)
    bin_lo=bin_lo.astype(np.int)
    bin_c=bin_c.astype(np.int)
    bin_size=bin_hi-bin_lo+1
    return (bin_lo,bin_hi,bin_c,bin_size)



