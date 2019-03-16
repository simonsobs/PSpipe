"""
@brief: python tools for analytical covariance matrix estimation.
"""
from __future__ import print_function
from copy import deepcopy
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
from pspy.cov_fortran import cov_fortran
import os,sys
import pickle 

def cov_coupling_spin0(win, lmax, niter=0,save_file=None):
    """
    @brief compute the coupling kernels corresponding to the T only covariance matrix of https://www.overleaf.com/read/fvrcvgbzqwrz
    @param win: the window functions, can be a so_map or a dictionnary containing so_map
                if the later, the entry of the dictionnary should be Ta,Tb,Tc,Td
    @param lmax: the maximum multipole to consider
    @param (optional) niter: the number of iteration performed while computing the alm
    @param (optional) save_file: the file in which the coupling kernel will be saved
    """
    coupling_dict={}
    if type(win) is not dict:
        sq_win=win.copy()
        sq_win.data*=sq_win.data
        alm= sph_tools.map2alm(sq_win,niter=niter,lmax=lmax)
        wcl= hp.alm2cl(alm)
        l=np.arange(len(wcl))
        wcl*=(2*l+1)/(4*np.pi)
        coupling=np.zeros((1,lmax,lmax))
        cov_fortran.calc_cov_spin0_single_win(wcl, coupling.T)
        coupling_dict['TaTcTbTd']=coupling[0]
        coupling_dict['TaTdTbTc']=coupling[0]
    else:
        wcl={}
        for s in ['TaTcTbTd','TaTdTbTc']:
            n0,n1,n2,n3=[s[i,i+2] for i in range(4)]
            sq_win_n0n1=win[n0].copy()
            sq_win_n0n1.data*=win[n1].data
            sq_win_n2n3=win[n2].copy()
            sq_win_n2n3.data*=win[n3].data
            alm_n0n1= sph_tools.map2alm(sq_win_n0n1,niter=niter,lmax=lmax)
            alm_n2n3= sph_tools.map2alm(sq_win_n2n3,niter=niter,lmax=lmax)
            wcl[n0+n1+n2+n3]= hp.alm2cl(alm_n0n1,alm_n2n3)
            l=np.arange(len(wcl[n0+n1+n2+n3]))
            wcl[n0+n1+n2+n3]*=(2*l+1)/(4*np.pi)
        
        coupling=np.zeros((2,lmax,lmax))
        cov_fortran.calc_cov_spin0(wcl['TaTcTbTd'],wcl['TaTdTbTc'], coupling.T)
        coupling_dict['TaTcTbTd']=coupling[0]
        coupling_dict['TaTdTbTc']=coupling[1]

    if save_file is not None:
        np.save('%s.npy'%save_file,coupling)

    return coupling_dict


def cov_coupling_spin0and2(win, lmax, niter=0,save_file=None):
    """
    @brief compute the coupling kernels corresponding to the T and E  covariance matrix of https://www.overleaf.com/read/fvrcvgbzqwrz
    @param win: the window functions, can be a so_map or a dictionnary containing so_map
    if the later, the entry of the dictionnary should be Ta,Tb,Tc,Td,Pa,Pb,Pc,Pd
    @param lmax: the maximum multipole to consider
    @param (optional) niter: the number of iteration performed while computing the alm
    @param (optional) save_file: the file in which the coupling kernel will be saved
    """
    win_list=['TaTcTbTd','TaTdTbTc','PaPcPbPd','PaPdPbPc','TaTcPbPd','TaPdPbTc','TaTcTbPd','TaPdTbTc','TaPcTbPd','TaPdTbPc','PaTcPbPd','PaPdPbTc']
    
    coupling_dict={}
    if type(win) is not dict:
        sq_win=win.copy()
        sq_win.data*=sq_win.data
        alm= sph_tools.map2alm(sq_win,niter=niter,lmax=lmax)
        wcl= hp.alm2cl(alm)
        l=np.arange(len(wcl))
        wcl*=(2*l+1)/(4*np.pi)
        coupling=np.zeros((3,lmax,lmax))
        cov_fortran.calc_cov_spin0and2_single_win(wcl, coupling.T)
        
        indexlist=[0,0,1,1,2,0,0,0,0,0,2,2]
        for name,index in zip(win_list,indexlist):
            coupling_dict[name]=coupling[index]
    else:
        wcl={}
        for s in win_list:
            n0,n1,n2,n3=[s[i,i+2] for i in range(4)]
            
            sq_win_n0n1=win[n0].copy()
            sq_win_n0n1.data*=win[n1].data
            sq_win_n2n3=win[n2].copy()
            sq_win_n2n3.data*=win[n3].data
            
            alm_n0n1= sph_tools.map2alm(sq_win_n0n1,niter=niter,lmax=lmax)
            alm_n2n3= sph_tools.map2alm(sq_win_n2n3,niter=niter,lmax=lmax)
            
            wcl[n0+n1+n2+n3]= hp.alm2cl(alm_n0n1,alm_n2n3)
            l=np.arange(len(wcl[n0+n1+n2+n3]))
            wcl[n0+n1+n2+n3]*=(2*l+1)/(4*np.pi)
    
        coupling=np.zeros((12,lmax,lmax))
        cov_fortran.calc_cov_spin0and2_single_win(wcl['TaTcTbTd'],wcl['TaTdTbTc'],wcl['PaPcPbPd'],wcl['PaPdPbPc'],wcl['TaTcPbPd'],wcl['TaPdPbTc'],wcl['TaTcTbPd'],
                                                  wcl['TaPdTbTc'],wcl['TaPcTbPd'],wcl['TaPdTbPc'],wcl['PaTcPbPd'],wcl['PaPdPbTc'],coupling.T)
            
        indexlist=np.arange(12)
        for name,index in zip(win_list,indexlist):
            coupling_dict[name]=coupling[index]

    if save_file is not None:
        np.save('%s.npy'%save_file,coupling)
    
    return coupling_dict


def read_coupling(file):
    """
    @brief read a precomputed coupling kernels
    the code use the size of the array to infer what type of survey it corresponds to
    """
    coupling=np.load('%s.npy'%file)
    coupling_dict={}
    if coupling.shape[0]==12:
        win_list=['TaTcTbTd','TaTdTbTc','PaPcPbPd','PaPdPbPc','TaTcPbPd','TaPdPbTc','TaTcTbPd','TaPdTbTc','TaPcTbPd','TaPdTbPc','PaTcPbPd','PaPdPbTc']
        indexlist=np.arange(12)
    elif coupling.shape[0]==3:
        win_list=['TaTcTbTd','TaTdTbTc','PaPcPbPd','PaPdPbPc','TaTcPbPd','TaPdPbTc','TaTcTbPd','TaPdTbTc','TaPcTbPd','TaPdTbPc','PaTcPbPd','PaPdPbTc']
        indexlist=[0,0,1,1,2,0,0,0,0,0,2,2]
    elif coupling.shape[0]==2:
        win_list=['TaTcTbTd','TaTdTbTc']
        indexlist=[0,1]
    elif coupling.shape[0]==1:
        win_list=['TaTcTbTd','TaTdTbTc']
        indexlist=[0,0]
    for name,index in zip(win_list,indexlist):
        coupling_dict[name]=coupling[index]
    
    return coupling_dict

def symmetrize(Clth,mode='arithm'):
    """
    @brief take a power spectrum Cl and return a symmetric array C_l1l2=f(Cl)
    @param mode (optional): geometric or arithmetic mean
    if geo return C_l1l2= sqrt( |Cl1 Cl2 |)
    if arithm return C_l1l2= (Cl1 + Cl2)/2
    default is arithmetic mean that can more easily deal with negative power spectrum
    """
    if mode=='geo':
        return np.sqrt(np.abs(np.outer(Clth,Clth)))
    if mode=='arithm':
        return np.add.outer(Clth, Clth)/2

def bin_mat(mat,binning_file,lmax):
    """
    @brief take a matrix and bin it Mbb'= Pbl Pb'l' Mll' with  Pbl =1/Nb sum_(l in b)
    @param binning_file: a binning file with format bin low, bin high, bin mean
    @param lmax: the maximum multipole to consider
    """
    bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    coupling_b=np.zeros((n_bins,n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            coupling_b[i,j]=np.mean(mat[bin_lo[i]:bin_hi[i],bin_lo[j]:bin_hi[j]])
    return coupling_b

def cov_spin0(Clth_dict,coupling_dict,binning_file,lmax,mbb_inv_ab,mbb_inv_cd):
    """
    @brief from the two point functions and the coupling kernel construct the spin0 analytical covariance matrix of <(C_ab- Clth)(C_cd-Clth)>
    @param Clth_dict: a dictionnary of theoretical power spectrum (auto and cross) for the different split combinaison ('TaTb' etc)
    @param coupling_dict: a dictionnary containing the coupling kernel
    @param binning_file: a binning file with format bin low, bin high, bin mean
    @param lmax: the maximum multipole to consider
    @param mbb_inv_ab: the inverse mode coupling matrix for the 'TaTb' power spectrum
    @param mbb_inv_cd: the inverse mode coupling matrix for the 'TcTd' power spectrum
    """
    cov=symmetrize(Clth_dict['TaTc'])*symmetrize(Clth_dict['TbTd'])*coupling_dict['TaTcTbTd']+ symmetrize(Clth_dict['TaTd'])*symmetrize(Clth_dict['TbTc'])*coupling_dict['TaTdTbTc']
    analytic_cov=bin_mat(cov,binning_file,lmax)
    analytic_cov=np.dot(np.dot(mbb_inv_ab,analytic_cov),mbb_inv_cd.T)
    return analytic_cov

def cov_spin0and2(Clth_dict,coupling_dict,binning_file,lmax,mbb_inv_ab,mbb_inv_cd):
    """
    @brief from the two point functions and the coupling kernel construct the T and E analytical covariance matrix of <(C_ab- Clth)(C_cd-Clth)>
    @param Clth_dict: a dictionnary of theoretical power spectrum (auto and cross) for the different split combinaison ('TaTb' etc)
    @param coupling_dict: a dictionnary containing the coupling kernel
    @param binning_file: a binning file with format bin low, bin high, bin mean
    @param lmax: the maximum multipole to consider
    @param mbb_inv_ab: the inverse mode coupling matrix for the 'ab' power spectra
    @param mbb_inv_cd: the inverse mode coupling matrix for the 'cd' power spectra
    """
    TaTc,TbTd,TaTd,TbTc=symmetrize(Clth_dict['TaTc']),symmetrize(Clth_dict['TbTd']),symmetrize(Clth_dict['TaTd']),symmetrize(Clth_dict['TbTc'])
    EaEc,EbEd,EaEd,EbEc=symmetrize(Clth_dict['EaEc']),symmetrize(Clth_dict['EbEd']),symmetrize(Clth_dict['EaEd']),symmetrize(Clth_dict['EbEc'])
    TaEd,TaEc,TbEc,TbEd,EaTc,EbTc=symmetrize(Clth_dict['TaEd']), symmetrize(Clth_dict['TaEc']), symmetrize(Clth_dict['TbEc']), symmetrize(Clth_dict['TbEd']), symmetrize(Clth_dict['EaTc']), symmetrize(Clth_dict['EbTc'])
    
    bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    analytic_cov=np.zeros((3*n_bins,3*n_bins))
    
    analytic_cov[:n_bins,:n_bins]=bin_mat(TaTc*TbTd*coupling_dict['TaTcTbTd']+ TaTd*TbTc*coupling_dict['TaTdTbTc'],binning_file,lmax) #TTTT
    analytic_cov[n_bins:2*n_bins,n_bins:2*n_bins]=bin_mat(TaTc*EbEd*coupling_dict['TaTcPbPd']+TaEd*EbTc*coupling_dict['TaPdPbTc'],binning_file,lmax) #TETE
    analytic_cov[2*n_bins:3*n_bins,2*n_bins:3*n_bins ]=bin_mat(EaEc*EbEd*coupling_dict['PaPcPbPd']+ EaEd*EbEc*coupling_dict['PaPdPbPc'],binning_file,lmax) #EEEE
    analytic_cov[n_bins:2*n_bins,:n_bins]=bin_mat(TaTc*TbEd*coupling_dict['TaTcTbPd']+TaEd*TbTc*coupling_dict['TaPdTbTc'],binning_file,lmax)  #TTTE
    analytic_cov[2*n_bins:3*n_bins,:n_bins]=bin_mat(TaEc*TbEd*coupling_dict['TaPcTbPd']+TaEd*TbEc*coupling_dict['TaPdTbPc'],binning_file,lmax) #TTEE
    analytic_cov[2*n_bins:3*n_bins,n_bins:2*n_bins]=bin_mat(EaTc*EbEd*coupling_dict['PaTcPbPd']+EaEd*EbTc*coupling_dict['TaPdTbPc'],binning_file,lmax) #TEEE
    
    analytic_cov = np.tril(analytic_cov) + np.triu(analytic_cov.T, 1)
    
    mbb_inv_ab=extract_TTTEEE_mbb(mbb_inv_ab)
    mbb_inv_cd=extract_TTTEEE_mbb(mbb_inv_cd)
    
    analytic_cov=np.dot(np.dot(mbb_inv_ab,analytic_cov),mbb_inv_cd.T)
    return analytic_cov

def extract_TTTEEE_mbb(mbb_inv):
    """
    @brief The mode coupling marix is computed for T,E,B but for now we only construct analytical covariance matrix for T and E
    The B modes is complex with important E->B leakage, this routine extract the T and E part of the mode coupling matrix
    @param mbb_inv: the inverse spin0 and 2 mode coupling matrix
    """
    mbb_inv_array=so_mcm.coupling_dict_to_array(mbb_inv)
    mbb_array=np.linalg.inv(mbb_inv_array)
    n_bins=int(mbb_array.shape[0]/9)
    mbb_array_select=np.zeros((3*n_bins,3*n_bins))
    mbb_array_select[:n_bins,:n_bins]=mbb_array[:n_bins,:n_bins]
    mbb_array_select[n_bins:2*n_bins,n_bins:2*n_bins]=mbb_array[n_bins:2*n_bins,n_bins:2*n_bins]
    mbb_array_select[2*n_bins:3*n_bins,2*n_bins:3*n_bins]=mbb_array[5*n_bins:6*n_bins,5*n_bins:6*n_bins]
    mbb_inv_array= np.linalg.inv(mbb_array_select)
    return mbb_inv_array

def cov2corr(cov):
    """
    @brief go from covariance to correlation, also setting the diagonal to zero
    @param cov: the covariance matrix
    """
    d = np.sqrt(cov.diagonal())
    corr = ((cov.T/d).T)/d - np.identity(cov.shape[0])
    return corr

def selectblock(cov, spectra,n_bins,block='TTTT'):
    """
    @brief select a block in a spin0 and 2 covariance matrix
    @param cov: the covariance matrix
    @param spectra: the arangement of the different block
    @param n_bins: the number of bins for each block
    @param block: the block you want to look at
    """
    if spectra==None:
        print ('cov mat of spin 0, no block selection needed')
        return
    else:
        blockindex={}
        for c1,s1 in enumerate(spectra):
            for c2,s2 in enumerate(spectra):
                blockindex[s1+s2]=[c1*n_bins,c2*n_bins]
    id1=blockindex[block][0]
    id2=blockindex[block][1]
    cov_select=cov[id1:id1+n_bins,id2:id2+n_bins]
    return cov_select

def delta(a,b):
    """
    @brief simple delta function
    """
    if a==b:
        return 1
    else:
        return 0

def calc_cov_lensed(noise_uK_arcmin, fwhm_arcmin, lmin, lmax, camb_lensed_theory_file, camb_unlensed_theory_file, output_dir, overwrite=False):
    """
    @brief wrapper around lenscov (https://github.com/JulienPeloton/lenscov). heavily borrowed from covariance.py
    compute lensing induced non-gaussian part of covariance matrix
    """
    try:
        import lib_covariances, lib_spectra, misc, util
    except:
        print("[ERROR] failed to load lenscov modules. Make sure that lenscov is properly installed")
        print("[ERROR] Note: lenscov is not yet python3 compatible")
    print("[WARNING] calc_cov_lensed requires MPI to be abled")
    from pspy import so_mpi
    from mpi4py import MPI
    so_mpi.init(True)

    rank, size = so_mpi.rank, so_mpi.size
    ## The available blocks in the code
    blocks     = ['TTTT','EEEE','BBBB','EEBB','TTEE','TTBB','TETE','TTTE','EETE','TEBB']

    ## Initialization of spectra
    cls_unlensed = lib_spectra.get_camb_cls(fname=os.path.abspath(camb_unlensed_theory_file), lmax=lmax)
    cls_lensed = lib_spectra.get_camb_cls(fname=os.path.abspath(camb_lensed_theory_file), lmax=lmax)
     
    file_manager = util.file_manager('covariances_CMBxCMB','pspy',spec='v1',lmax=lmax,
            force_recomputation=overwrite,folder=output_dir,rank=rank)

    if file_manager.FileExist is True:
        if rank == 0:
            print('Already computed in %s/' %output_dir)
    else:
        cov_order0_tot, cov_order1_tot, cov_order2_tot, junk = lib_covariances.analytic_covariances_CMBxCMB(
            cls_unlensed,
            cls_lensed,
            lmin=lmin,
            blocks=blocks,
            noise_uK_arcmin=noise_uK_arcmin,
            TTcorr=False,
            fwhm_arcmin=fwhm_arcmin,
            MPI=MPI,
            use_corrfunc=True,
            exp='pspy',
            folder_cache=output_dir)
        array_to_save = [cov_order0_tot, cov_order1_tot, cov_order2_tot, blocks]

        if file_manager.FileExist is False and rank == 0:
            file_manager.save_data_on_disk(array_to_save)

def load_cov_lensed(cov_lensed_file, include_gaussian_part=False):
    """
    @brief wrapper around lenscov (https://github.com/JulienPeloton/lenscov). 
    @param include_gaussian_part: if False, it returns only lensing induced non-gaussin parts
    """
    covs = {}

    input_data = pickle.load(open(cov_lensed_file, 'r'))['data']
    cov_G      = input_data[0]
    cov_NG1    = input_data[1]
    cov_NG2    = input_data[2]
    combs      = input_data[3]

    for ctr, comb in enumerate(combs):
        covs[comb] = cov_NG1[ctr].data + cov_NG2[ctr].data
        if include_gaussian_part:
            covs[comb] = covs[comb] + cov_G[ctr].data

    return covs

