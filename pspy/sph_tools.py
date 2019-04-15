"""
@brief: python routines for generalized map2alm and alm2map.
"""
from __future__ import absolute_import, print_function
from pixell import curvedsky,powspec
import healpy as hp, pylab as plt, numpy as np, astropy.io.fits as pyfits
import sys,os,copy


def map2alm(map,niter,lmax,theta_range=None):
    """
    @brief general map2alm transform
    @param map: a so map object
    @param niter: the number of iteration performed while computing the alm
    @param lmax: the maximum multipole of the transform
    @param theta_range: for healpix pixellisation you can specify
    a range [theta_min,theta_max] in radian. All pixel outside this range
    will be assumed to be zero.
    """
    print ('niter',niter)
    if map.pixel=='HEALPIX':
        if theta_range is None:
            alm= hp.sphtfunc.map2alm(map.data,lmax=lmax,iter=niter)
        
        else:
            nside=hp.pixelfunc.get_nside(map.data)
            alm= curvedsky.map2alm_healpix(map.data,lmax=lmax,theta_min=theta_range[0], theta_max=theta_range[1])
            if niter !=0:
                map_copy=map.copy()
                alm= curvedsky.map2alm_healpix(map.data,lmax=lmax,theta_min=theta_range[0], theta_max=theta_range[1])
                for k in range(niter):
                    alm += curvedsky.map2alm_healpix(map.data-curvedsky.alm2map_healpix(alm,map_copy.data),lmax=lmax,theta_min=thetas[0], theta_max=thetas[1])
        return alm

    elif map.pixel=='CAR':
        alm = curvedsky.map2alm(map.data,lmax= lmax)
        if niter !=0:
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

def get_alms(map,window,niter,lmax,theta_range=None):
    """
    @brief get a map, multiply by a window and return alm
    @param map: so map containing the data
    @param windw: a so map with the window function, if the so map has 3 components
    (for spin0 and 2 fields) expect a tuple (window,window_pol)
    @param (optional) theta range: for healpix pixellisation you can specify
    a range [theta_min,theta_max] in radian. All pixel outside this range
    will be assumed to be zero.
    @return: the alms
    """
    windowed_map=map.copy()
    if map.ncomp ==3:
        windowed_map.data[0]=map.data[0]*window[0].data
        windowed_map.data[1]=map.data[1]*window[1].data
        windowed_map.data[2]=map.data[2]*window[1].data
    if map.ncomp ==1:
        windowed_map.data=map.data*window.data
    alms=map2alm(windowed_map,niter,lmax,theta_range=theta_range)
    return alms


def get_pure_alms(map,window,niter,lmax):
    
    s1_a,s1_b,s2_a,s2_b=get_spinned_windows(window[1],lmax,niter=niter)
    p2 = np.array([window[1].data*map.data[1], window[1].data*map.data[2]])
    p1 = np.array([(s1_a.data*map.data[1] + s1_b.data*map.data[2]), (s1_a.data*map.data[2] - s1_b.data*map.data[1])])
    p0 = np.array([(s2_a.data*map.data[1] + s2_b.data*map.data[2]), (s2_a.data*map.data[2] - s2_b.data*map.data[1])])
    
    if map.pixel=='CAR':
        p0=enmap.samewcs(p0,map.data)
        p1=enmap.samewcs(p1,map.data)
        p2=enmap.samewcs(p2,map.data)
        
        alm=curvedsky.map2alm(map.data[0]*window[0].data,lmax= lmax)
        s2eblm = curvedsky.map2alm(p2,spin=2,lmax= lmax)
        s1eblm = curvedsky.map2alm(p1,spin=1,lmax= lmax)
        s0eblm= s1eblm.copy()
        s0eblm[0] = curvedsky.map2alm(p0[0],spin=0,lmax= lmax)
        s0eblm[1] = curvedsky.map2alm(p0[1],spin=0,lmax= lmax)
    
    if map.pixel=='HEALPIX':
        alm=hp.sphtfunc.map2alm(map.data[0]*window[0].data,lmax=lmax,iter=niter)#curvedsky.map2alm_healpix(map.data[0]*window[0].data,lmax= lmax)
        s2eblm = curvedsky.map2alm_healpix(p2,spin=2,lmax= lmax)
        s1eblm = curvedsky.map2alm_healpix(p1,spin=1,lmax= lmax)
        s0eblm= s1eblm.copy()
        s0eblm[0] = curvedsky.map2alm_healpix(p0[0],spin=0,lmax= lmax)
        s0eblm[1] = curvedsky.map2alm_healpix(p0[1],spin=0,lmax= lmax)

    ell = np.arange(lmax)
    filter_1=np.zeros(lmax)
    filter_2=np.zeros(lmax)
    filter_3=np.zeros(lmax)

    filter_1[2:]=2*np.sqrt(1.0 /((ell[2:] + 2.)*(ell[2:] - 1.)))
    filter_2[2:]= np.sqrt(1.0 /((ell[2:] + 2.)*(ell[2:] + 1.)*ell[2:]*(ell[2:] - 1.)))
    filter_3[2:]= ell[2:]*0+ 1
    for k in range(2):
        s1eblm[k]=hp.almxfl(s1eblm[k],filter_1)
        s0eblm[k]=hp.almxfl(s0eblm[k],filter_2)
        s2eblm[k]=hp.almxfl(s2eblm[k],filter_3)
    
    elm_p = s2eblm[0] + s1eblm[0] - s0eblm[0]
    blm_b = s2eblm[1] + s1eblm[1] - s0eblm[1]
    
    return np.array([alm,elm_p,blm_b])

