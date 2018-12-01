"""
@brief: python routines for mode coupling calculation.
"""
import healpy as hp, pylab as plt, numpy as np
from pspy import sph_tools
import mcm_fortran
from copy import deepcopy
import pspy_utils


def mcm_and_bbl_spin0(win1, binning_file, lmax, type, win2=None,bl1=None,bl2=None,input_alm=False,niter=0,return_mcm=False):
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

    if input_alm==False:
        win1= sph_tools.map2alm(win1,niter=niter,lmax=lmax)
        if win2 is not None:
            win2= sph_tools.map2alm(win2,niter=niter,lmax=lmax)

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

    mcm=np.zeros((lmax,lmax))
    mcm_fortran.calc_mcm_spin0(wcl, bl1*bl2,mcm.T)
    bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    mbb=np.zeros((n_bins,n_bins))
    mcm_fortran.bin_mcm(mcm.T, bin_lo,bin_hi,bin_size, mbb.T,doDl)
    Bbl=np.zeros((n_bins,lmax))
    mcm_fortran.binning_matrix(mcm.T,bin_lo,bin_hi,bin_size, Bbl.T,doDl)
    Bbl=np.dot(np.linalg.inv(mbb),Bbl)

    if return_mcm:
        return mcm,mbb,Bbl
    else:
        return mbb, Bbl

def mcm_and_bbl_spin0and2(win1, binning_file,lmax,type='Dl', win2=None, bl1=None,bl2=None,input_alm=False,niter=0,pureB=False,return_mcm=False):
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

    if input_alm==False:
        win1= (sph_tools.map2alm(win1[0],niter=niter,lmax=lmax), sph_tools.map2alm(win1[1],niter=niter,lmax=lmax))
        if win2 is not None:
            win2= (sph_tools.map2alm(win2[0],niter=niter,lmax=lmax), sph_tools.map2alm(win2[1],niter=niter,lmax=lmax))
    if win2 is None:
        win2=deepcopy(win1)

    if bl1 is None:
        bl1=(np.ones(lmax),np.ones(lmax))
    if bl2 is None:
        bl2=deepcopy(bl1)

    wcl={}
    wbl={}
    spin=['0','2']
    for i,s1 in enumerate(spin):
        for j,s2 in enumerate(spin):
            wcl[s1+s2]=hp.alm2cl(win1[i],win2[i])
            wcl[s1+s2]=wcl[s1+s2][:lmax]*(2*np.arange(lmax)+1)
            wbl[s1+s2]=bl1[i]*bl2[j]

    mcm=np.zeros((5,lmax,lmax))

    if pureB==False:
        mcm_fortran.calc_mcm_spin0and2(wcl['00'],wcl['02'],wcl['20'],wcl['22'], wbl['00'],wbl['02'],wbl['20'], wbl['22'],mcm.T)
    else:
        #mcm_fortran.calc_mcm_spin0and2_pureB(wcl['00'],wcl['02'],wcl['20'],wcl['22'], wbl['00'],wbl['02'],wbl['20'], wbl['22'],mcm.T)
        print 'not implemented yet'

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

    spin_pair=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']
    for s in spin_pair:
        Bbl[s]=np.dot(np.linalg.inv(mbb[s]),Bbl[s])

    if return_mcm:
        return mcm,mbb,Bbl
    else:
        return mbb, Bbl

def dict_to_array(dict):
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

def apply_Bbl(Bbl,ps,fields=None):
    """
    @brief bin theoretical power spectra
    @param Bbl: a binning matrix, if fields is not None will be a Bbl dictionnary, otherwise a (n_bins,lmax) matrix
    @param ps: a theoretical power spectrum: if fields is not None will be a ps dictionnary, otherwise a (lmax) vector
    @return Dbth: a theoretical binned power spectrum
    """
    if fields is not None:
        Bbl_array=dict_to_array(Bbl)
        ps_array=ps[fields[0]]
        for f in fields[1:]:
            print ps_array.shape
            ps_array=np.append(ps_array, ps[f])
        ps_b=np.dot(Bbl_array,ps_array)
        n_bins=Bbl_array.shape[0]/9
        ps_th={}
        for i,f in enumerate(fields):
            ps_th[f]=ps_b[i*n_bins:(i+1)*n_bins]
        return ps_th
    else:
        ps_th=np.dot(Bbl,ps)
    return ps_th



