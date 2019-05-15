"""
@brief: python routines for mode coupling calculation.
"""
from __future__ import absolute_import, print_function
import healpy as hp, pylab as plt, numpy as np
from pspy import sph_tools
from pspy.mcm_fortran import mcm_fortran
from copy import deepcopy
from pspy import pspy_utils


def mcm_and_bbl_spin0(win1, binning_file, lmax,niter, type, win2=None,bl1=None,bl2=None,input_alm=False,unbin=None,save_file=None,lmax_pad=None):
    """
    @brief get the mode coupling matrix and the binning matrix for spin0 fields
    @param win1: the window function of survey 1, if input_alm=True, expect wlm1
    @param binning_file: a binning file with format bin low, bin high, bin mean
    @param lmax: the maximum multipole to consider
    @param type: the type of binning, either bin Cl or bin Dl
    @param win2: optional, the window function of survey 2, if input_alm=True, expect wlm2
    @param bl1: optional, the beam of survey 1
    @param bl2: optional, the beam of survey 2
    @param niter: optional, if input_alm=False, specify the number of iteration in map2alm
    @param return_mcm: optional, return the unbinned mode coupling matrix
    """
    if type=='Dl':
        doDl=1
    if type=='Cl':
        doDl=0

    maxl=lmax
    if lmax_pad is not None:
        maxl=lmax_pad

    if input_alm==False:
        win1= sph_tools.map2alm(win1,niter=niter,lmax=maxl)
        if win2 is not None:
            win2= sph_tools.map2alm(win2,niter=niter,lmax=maxl)

    if win2 is None:
        wcl= hp.alm2cl(win1)
    else:
        wcl= hp.alm2cl(win1,win2)

    l=np.arange(len(wcl))
    wcl*=(2*l+1)

    if bl1 is None:
        bl1=np.ones(len(l))
    if bl2 is None:
        bl2= bl1.copy()

    mcm=np.zeros((maxl,maxl))
    mcm_fortran.calc_mcm_spin0(wcl, bl1*bl2,mcm.T)
    mcm=mcm[:lmax,:lmax]
    bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    mbb=np.zeros((n_bins,n_bins))
    mcm_fortran.bin_mcm(mcm.T, bin_lo,bin_hi,bin_size, mbb.T,doDl)
    if unbin:
        mcm_inv=np.linalg.inv(mcm)

    Bbl=np.zeros((n_bins,lmax))
    mcm_fortran.binning_matrix(mcm.T,bin_lo,bin_hi,bin_size, Bbl.T,doDl)
    mbb_inv= np.linalg.inv(mbb)
    Bbl=np.dot(mbb_inv,Bbl)

    if unbin:
        if save_file is not None:
            save_coupling(save_file,mbb_inv,Bbl,mcm_inv=mcm_inv)
        return mcm_inv,mbb_inv,Bbl
    else:
        if save_file is not None:
            save_coupling(save_file,mbb_inv,Bbl)
        return mbb_inv, Bbl

def mcm_and_bbl_spin0and2(win1, binning_file,lmax,niter,type='Dl', win2=None, bl1=None,bl2=None,input_alm=False,pure=False,unbin=None,save_file=None,lmax_pad=None):
    """
    @brief get the mode coupling matrix and the binning matrix for spin 0 and 2 fields
    @param win1: a python tuple (win_spin0,win_spin2) with the window functions of survey 1, if input_alm=True, expect (wlm_spin0, wlm_spin2)
    @param binning_file: a binning file with format bin low, bin high, bin mean
    @param lmax: the maximum multipole to consider
    @param type: the type of binning, either bin Cl or bin Dl
    @param win2: optional, a python tuple (win_spin0,win_spin2) with the window functions of survey 2, if input_alm=True, expect expect (wlm_spin0, wlm_spin2)
    @param bl1: optional,  a python tuple (beam_spin0,beam_spin2) with the beam of survey 1
    @param bl2: optional,  a python tuple (beam_spin0,beam_spin2) with the beam of survey 2
    @param niter: optional, if input_alm=False, specify the number of iteration in map2alm
    @param pureB: optional, do B mode purification
    @param return_mcm: optional, return the unbinned mode coupling matrix
    """
    
    def get_coupling_dict(array,fac=1.0):
        ncomp,dim1,dim2=array.shape
        dict={}
        dict['spin0xspin0']=array[0,:,:]
        dict['spin0xspin2']=array[1,:,:]
        dict['spin2xspin0']=array[2,:,:]
        dict['spin2xspin2']=np.zeros((4*dim1,4*dim2))
        for i in range(4):
            dict['spin2xspin2'][i*dim1:(i+1)*dim1,i*dim2:(i+1)*dim2]=array[3,:,:]
        dict['spin2xspin2'][2*dim1: 3*dim1,dim2:2*dim2]=array[4,:,:]*fac
        dict['spin2xspin2'][dim1:2*dim1,2*dim2: 3*dim2]=array[4,:,:]*fac
        dict['spin2xspin2'][3*dim1: 4*dim1,:dim2]=array[4,:,:]
        dict['spin2xspin2'][:dim1,3*dim2:4*dim2]=array[4,:,:]
        return dict
    
    if type=='Dl':
        doDl=1
    if type=='Cl':
        doDl=0

    maxl=lmax
    if lmax_pad is not None:
        maxl=lmax_pad

    if input_alm==False:
        win1= (sph_tools.map2alm(win1[0],niter=niter,lmax=maxl), sph_tools.map2alm(win1[1],niter=niter,lmax=maxl))
        if win2 is not None:
            win2= (sph_tools.map2alm(win2[0],niter=niter,lmax=maxl), sph_tools.map2alm(win2[1],niter=niter,lmax=maxl))
    if win2 is None:
        win2=deepcopy(win1)

    if bl1 is None:
        bl1=(np.ones(maxl),np.ones(maxl))
    if bl2 is None:
        bl2=deepcopy(bl1)

    wcl={}
    wbl={}
    spin=['0','2']

    for i,s1 in enumerate(spin):
        for j,s2 in enumerate(spin):
            wcl[s1+s2]=hp.alm2cl(win1[i],win2[j])
            #wcl[s1+s2]=wcl[s1+s2][:lmax]*(2*np.arange(lmax)+1)
            wcl[s1+s2]=wcl[s1+s2]*(2*np.arange(len(wcl[s1+s2]))+1)

            wbl[s1+s2]=bl1[i]*bl2[j]

    mcm=np.zeros((5,maxl,maxl))

    if pure==False:
        mcm_fortran.calc_mcm_spin0and2(wcl['00'],wcl['02'],wcl['20'],wcl['22'], wbl['00'],wbl['02'],wbl['20'], wbl['22'],mcm.T)
    else:
        mcm_fortran.calc_mcm_spin0and2_pure(wcl['00'],wcl['02'],wcl['20'],wcl['22'], wbl['00'],wbl['02'],wbl['20'], wbl['22'],mcm.T)

    mcm=mcm[:,:lmax,:lmax]

    bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)

    mbb_array=np.zeros((5,n_bins,n_bins))
    Bbl_array=np.zeros((5,n_bins,lmax))

    for i in range(5):
        mcm_fortran.bin_mcm((mcm[i,:,:]).T, bin_lo,bin_hi,bin_size, (mbb_array[i,:,:]).T,doDl)
        mcm_fortran.binning_matrix((mcm[i,:,:]).T,bin_lo,bin_hi,bin_size, (Bbl_array[i,:,:]).T,doDl)

    mcm= get_coupling_dict(mcm,fac=-1.0)
    mbb= get_coupling_dict(mbb_array,fac=-1.0)
    Bbl= get_coupling_dict(Bbl_array,fac=1.0)

    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']
    mbb_inv={}
    mcm_inv={}
    for s in spin_pairs:
        if unbin:
            mcm_inv[s]=np.linalg.inv(mcm[s])
        mbb_inv[s]=np.linalg.inv(mbb[s])
        Bbl[s]=np.dot(mbb_inv[s],Bbl[s])

    if unbin:
        if save_file is not None:
            save_coupling(save_file,mbb_inv,Bbl,spin_pairs=spin_pairs,mcm=mcm_inv)
        return mcm_inv,mbb_inv,Bbl
    else:
        if save_file is not None:
            save_coupling(save_file,mbb_inv,Bbl,spin_pairs=spin_pairs)
        return mbb_inv,Bbl

def coupling_dict_to_array(dict):
    """
    @brief take a mcm or Bbl dictionnary with entries:
    -  (spin0xspin0)
    -  (spin0xspin2)
    -  (spin2xspin0)
    -  (spin2xspin2)
    and return a 9xdim1,9xdim2 array
    """

    dim1,dim2=dict['spin0xspin0'].shape
    array=np.zeros((9*dim1,9*dim2))
    array[0:dim1,0:dim2]=dict['spin0xspin0']
    array[dim1:2*dim1,dim2:2*dim2]=dict['spin0xspin2']
    array[2*dim1:3*dim1,2*dim2:3*dim2]=dict['spin0xspin2']
    array[3*dim1:4*dim1,3*dim2:4*dim2]=dict['spin2xspin0']
    array[4*dim1:5*dim1,4*dim2:5*dim2]=dict['spin2xspin0']
    array[5*dim1:9*dim1,5*dim2:9*dim2]=dict['spin2xspin2']
    return array

def apply_Bbl(Bbl,ps,spectra=None):
    """
    @brief bin theoretical power spectra
    @param Bbl: a binning matrix, if spectra is not None will be a Bbl dictionnary for spin0 and 2 fields, otherwise a (n_bins,lmax) matrix
    @param ps: a theoretical power spectrum: if spectra is not None will be a ps dictionnary, otherwise a (lmax) vector
    @param (optional) spectra,  needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    @return Dbth: a theoretical binned power spectrum
    """
    if spectra is not None:
        Bbl_array=coupling_dict_to_array(Bbl)
        ps_vec=ps[spectra[0]]
        for f in spectra[1:]:
            ps_vec=np.append(ps_vec, ps[f])
        ps_b=np.dot(Bbl_array,ps_vec)
        n_bins=int(Bbl_array.shape[0]/9)
        ps_th={}
        for i,f in enumerate(spectra):
            ps_th[f]=ps_b[i*n_bins:(i+1)*n_bins]
        return ps_th
    else:
        ps_th=np.dot(Bbl,ps)
    return ps_th

def save_coupling(prefix,mbb_inv,Bbl,spin_pairs=None,mcm_inv=None):
    """
    @brief save the inverse of the mode coupling matrix and the binning matrix in npy format
    @param prefix: the prefix for the name of the file
    @param mbb_inv: the inverse of the mode coupling matrix, if spin pairs is not none, should be a dictionnary with entries
    -  (spin0xspin0)
    -  (spin0xspin2)
    -  (spin2xspin0)
    -  (spin2xspin2)
    otherwise, it will be a single matrix
    @param Bbl,  the binning matrix,if spin pairs is not none, should be a dictionnary with entries
    -  (spin0xspin0)
    -  (spin0xspin2)
    -  (spin2xspin0)
    -  (spin2xspin2)
    otherwise, it will be a single matrix
    @param (optional) spin_pairs: needed for spin0 and 2 fields.
    """

    if spin_pairs is not None:
        for s in spin_pairs:
            np.save(prefix +'_mbb_inv_%s.npy'%s,mbb_inv[s])
            np.save(prefix +'_Bbl_%s.npy'%s,Bbl[s])
            if mcm_inv is not None:
                np.save(prefix +'_mcm_inv_%s.npy'%s,mcm_inv[s])
    else:
        np.save(prefix +'_mbb_inv.npy',mbb_inv)
        np.save(prefix +'_Bbl.npy',Bbl)
        if mcm_inv is not None:
            np.save(prefix +'_mcm_inv.npy',mcm_inv)


def read_coupling(prefix,spin_pairs=None,unbin=None):
    """
    @brief read the inverse of the mode coupling matrix and the binning matrix
    @param prefix: the prefix for the name of the file
    @param (optional) spin_pairs: needed for spin0 and 2 fields.
    """

    if spin_pairs is not None:
        Bbl={}
        mbb_inv={}
        mcm={}
        for s in spin_pairs:
            if unbin:
                mcm[s]= np.load(prefix+'_mcm_inv_%s.npy'%s)
            mbb_inv[s]= np.load(prefix+'_mbb_inv_%s.npy'%s)
            Bbl[s]= np.load(prefix+'_Bbl_%s.npy'%s)
    else:
        if unbin:
            mcm= np.load(prefix+'_mcm_inv.npy')
        mbb_inv=np.load(prefix +'_mbb_inv.npy')
        Bbl=np.load(prefix +'_Bbl.npy')

    if unbin:
        return mcm,mbb_inv,Bbl
    else:
        return mbb_inv,Bbl
