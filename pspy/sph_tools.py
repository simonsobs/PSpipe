from __future__ import print_function
from pixell import curvedsky,powspec
import healpy as hp, pylab as plt, numpy as np, astropy.io.fits as pyfits
import sys,os,copy


def map2alm(map,niter,lmax=None,theta_range=None):
    """
    @brief general map2alm transform
    @param map: a so map object
    @param niter: the number of iteration performed while computing the alm
    @param lmax: the maximum multipole of the transform
    @param theta_range: for healpix pixellisation you can specify
    a range [theta_min,theta_max] in radian. All pixel outside this range
    will be assumed to be zero.
    """
    if map.pixel=='HEALPIX':
        if theta_range is None:
            alm= hp.sphtfunc.map2alm(map.data,lmax=lmax,iter=niter)
        else:
            nside=hp.pixelfunc.get_nside(map.data)
            alm= curvedsky.map2alm_healpix(map.data,lmax=lmax,theta_min=theta_range[0], theta_max=theta_range[1])
            if iter !=0:
                map_copy=map.copy()
                alm= curvedsky.map2alm_healpix(map.data,lmax=lmax,theta_min=theta_range[0], theta_max=theta_range[1])
                for k in range(niter):
                    alm += curvedsky.map2alm_healpix(map.data-curvedsky.alm2map_healpix(alm,map_copy.data),lmax=lmax,theta_min=thetas[0], theta_max=thetas[1])
        return alm

    elif map.pixel=='CAR':
        alm = curvedsky.map2alm(map.data,lmax= lmax)
        if iter !=0:
            map_copy=map.copy()
            for k in range(niter):
                alm += curvedsky.map2alm(map.data-curvedsky.alm2map(alm,map_copy.data),lmax=lmax)
    else:
        print ('Error: file %s is neither a enmap or a healpix map'%file)
        sys.exit()

    alm = alm.astype(np.complex128)
    return alm


def alm2map(alms,map):
    """
    @brief general alm2map transform
    @param alms: a set of alms, the shape of alms should correspond to map.ncomp
    @param map: a so map object
    @return: a so map instance with value given by the alms
    """
    
    if map.ncomp==1:
        spin=0
    else:
        spin=[0,2]
    if map.pixel=='HEALPIX':
        map.data=curvedsky.alm2map_healpix(alms,map.data,spin)
    elif map.pixel=='CAR':
        map.data=curvedsky.alm2map(alms,map.data,spin)
    else:
        print ('Error: file %s is neither a enmap or a healpix map'%file)
        sys.exit()
    return map


def read_cls(clfile,lmax=None):
    """
    @brief take CAMB lensed power spectrum and return a cl dictionnary
    """
    fields=['TT','EE','BB','TE']
    cl={}
    lth, cl['TT'], cl['EE'], cl['BB'], cl['TE'] = np.loadtxt(clfile,unpack=True)
    fth=lth*(lth+1)/(2*np.pi)
    for l1 in fields:
        cl[l1]/=fth
        cl[l1][0]=0
    if lmax is not None:
        lth=lth[:lmax]
        for l1 in fields:
            cl[l1]=cl[l1][:lmax]
    cl['ET']=cl['TE']
    cl['TB']=cl['TT']*0
    cl['BT']=cl['TT']*0
    cl['EB']=cl['TT']*0
    cl['BE']=cl['TT']*0
    return(lth,cl)




