"""
@brief: python routines for mode coupling calculation.
Should include an option for pure B mode estimation
"""
import healpy as hp, pylab as plt, numpy as np
import mcm_code_thibaut
# import mcm_code_dw
# import mcm_code_steve

def mcm_and_bbl_spin0_thibaut(wlm1, binning_file,lmax, wlm2=None,bl1=None,bl2=None,type='Dl'):
    """
    @brief Thibaut's version to get the mode coupling matrix and the binning matrix for spin0 fields
    @param wlm1: the harmonic transform of the window of survey 1
    @param wlm2: optional, the harmonic transform of the window of survey 2
    @param bl1: optional, the beam of survey 1
    @param bl2: optional, the beam of survey 2
    @param binning_file: a binning file with format bin low, bin high, bin mean
    @param lmax: the maximum multipole to consider
    @param type: the type of binning, either bin Cl or bin Dl
    """
    
    def read_binning_file(file,lmax):
        binLo,binHi,binC = plt.loadtxt(file,unpack=True)
        id = np.where(binHi <lmax)
        binLo,binHi,binC=binLo[id],binHi[id],binC[id]
        if binLo[0]<2:
            binLo[0]=2
        binHi=binHi.astype(np.int)
        binLo=binLo.astype(np.int)
        binSize=binHi-binLo+1
        return (binLo,binHi,binSize)
    
    if type=='Dl':
        doDl=1
    if type=='Cl':
        doDl=0

    if wlm2 is None:
        wcl= hp.alm2cl(wlm1)
    else:
        wcl= hp.alm2cl(wlm1,wlm2)

    l=np.arange(len(wcl))
    wcl*=(2*l+1)

    if bl1 is None:
        bl1=np.ones(len(l))
    if bl2 is None:
        bl2= bl1.copy()

    mcm=np.zeros((lmax,lmax))
    mcm_code_thibaut.calc_mcm(wcl, bl1*bl2,mcm.T)
    bin_lo,bin_hi,bin_size= read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    mbb=np.zeros((n_bins,n_bins))
    mcm_code_thibaut.bin_mcm(mcm.T, bin_lo,bin_hi,bin_size, mbb.T,doDl)
    Bbl=np.zeros((n_bins,lmax))
    mcm_code_thibaut.binning_matrix(mcm.T,bin_lo,bin_hi,bin_size, Bbl.T,doDl)
    Bbl=np.dot(np.linalg.inv(mbb),Bbl)

    return mbb, Bbl







