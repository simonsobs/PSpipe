"""
@brief: utils for pspy.
"""
import healpy as hp, pylab as plt, numpy as np

def ps_lensed_theory_to_dict(filename,lmax,type,spin0=False):
    """
    @brief read a lensed power spectrum from CAMB and return a dictionnary
    @param filename: the name of the CAMB lensed power spectrum
    @param lmax: the maximum multipole
    @param type: 'Cl' or 'Dl'
    @param optional: spin0, if true only return ps['TT']
    @return ps: a dictionnary file with the power spectra
    """
    fields=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    ps={}
    l,ps['TT'],ps['EE'],ps['BB'],ps['TE']=np.loadtxt(filename,unpack=True)
    ps['ET']=ps['TE']
    ps['TB'],ps['BT'],ps['EB'],ps['BE']=np.zeros((4,len(l)))
    
    scale=l*(l+1)/(2*np.pi)

    for f in fields:
        ps[f]=ps[f][:lmax]
        if type=='Cl':
            ps[f]/=scale[:lmax]
    if spin0==True:
        return l,ps['TT']
    else:
        return l,ps

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



