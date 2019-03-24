import numpy as np
import pyfftw
import os
import multiprocessing
from pixell import utils

def sin_profile(N,N_cut,rising=False):
    """ gives 1 dimensinal profile with sin
        total length N
        sin profile starting at N_cut
        """
    if rising:
        sin_apo = np.zeros(N)
        for i in range(int(N)):
            if i>=N_cut:
                sin_apo[i] = 1-(-1.0/(2.0*np.pi)*np.sin(2.0*np.pi*(N-i)/float(N-N_cut))+(N-i)/float(N-N_cut))
    else:
        sin_apo = np.ones(int(N))
        for i in range(int(N)):
            if i>=N_cut:
                sin_apo[i] = (-1.0/(2.0*np.pi)*np.sin(2.0*np.pi*(N-i)/float(N-N_cut))+(N-i)/float(N-N_cut))
    return sin_apo


def get_kcut_profile(kmax,kcut,apo_over):
    """ gives 1 dimensinal apodization profile with sin
        kmax is the total length
        0 for <kcut
        apodization happens over apo_over
        """
    ret = np.ones(kmax)
    if kcut+apo_over == 0:
        return ret
    
    rising = sin_profile(kcut+apo_over,kcut,1)
    falling = 1.0-sin_profile(kcut+apo_over,kcut,0)[::-1]
    
    ret[:len(rising)]=rising
    if len(falling) == 0:
        print ("fourier resolution bigger than the cut size asked")
        pass
    else:
        ret[-len(falling):]=falling
    
    ret[ret<0] = 0
    return np.fft.fftshift(ret)

def gen_kx_mask(m,d_th,kx_cut,kx_cut_apo):
    """ given 2d fourier map and resolution (pixel size d_th),
        return a kx mask with lx cut at kx_cut and apodize over
        kx_cut_apo
        """
    if kx_cut == 0:
        return np.ones_like(m)
    
    ell_scale_factor = 2.*np.pi/(d_th*np.pi/180.)
    
    Ny,Nx = np.real(m).shape
    
    inds  = [(np.arange(Nx)+.5 - Nx/2.) /(Nx-1.)]
    X = np.repeat(inds,Ny,axis=0) * ell_scale_factor
    
    cut_ind = np.where(np.abs(np.fft.fftshift(X)[0]-kx_cut)==np.min(np.abs(np.fft.fftshift(X)[0]-kx_cut)))[0]
    apo_ind = np.where(np.abs(np.fft.fftshift(X)[0]-(kx_cut+kx_cut_apo))==np.min(np.abs(np.fft.fftshift(X)[0]-(kx_cut+kx_cut_apo))))[0]
    
    return np.repeat([get_kcut_profile(Nx,cut_ind,apo_ind-cut_ind)],Ny,axis=0)


def gen_ky_mask(m,d_th,ky_cut,ky_cut_apo):
    """ given 2d fourier map and resolution (pixel size d_th),
        return a ky mask with ly cut at ky_cut and apodize over
        ky_cut_apo
        """
    if ky_cut == 0:
        return np.ones_like(m)
    
    ell_scale_factor = 2.*np.pi/(d_th*np.pi/180.)
    
    Ny,Nx = np.real(m).shape
    
    inds  = (np.arange(Ny)+.5 - Ny/2.) /(Ny-1.)
    Y = np.repeat(inds.reshape(Ny,1),Nx,axis=1) * ell_scale_factor
    
    cut_ind = np.where(np.abs(np.fft.fftshift(Y)[:,0]-ky_cut)==np.min(np.abs(np.fft.fftshift(Y)[:,0]-ky_cut)))[0]
    apo_ind = np.where(np.abs(np.fft.fftshift(Y)[:,0]-(ky_cut+ky_cut_apo))==np.min(np.abs(np.fft.fftshift(Y)[:,0]-(ky_cut+ky_cut_apo))))[0]
    
    return np.repeat(get_kcut_profile(Ny,cut_ind,apo_ind-cut_ind).reshape(Ny,1),Nx,axis=1)


def calc_window(shape):
    """from enlib import enmap"""
    """Compute fourier-space window function. Like the other fourier-based
        functions in this module, equi-spaced pixels are assumed. Since the
        window function is separable, it is returned as an x and y part,
        such that window = wy[:,None]*wx[None,:]."""
    wy = np.sinc(np.fft.fftfreq(shape[-2]))
    wx = np.sinc(np.fft.fftfreq(shape[-1]))
    return wy, wx

def get_map_kx_ky_filtered_pyfftw(map,apo,filter_dict):
    """ given input m apply a 2d fourier mask with ky_cut and apodization over ky_cut_apo, then return the filtered map
        uses pyfftw for fft's
        """
    try:
        ncore = int(os.environ['OMP_NUM_THREADS'])
    except (KeyError, ValueError):
        ncore = multiprocessing.cpu_count()

    if map.ncomp==1:
        map.data=apply_filter(map.data,apo.data,filter_dict,ncore)
    else:
        for i in range(map.ncomp):
            map.data[i]=apply_filter(map.data[i],apo.data,filter_dict,ncore)

    return map

def apply_filter(comp,apo,filter_dict,ncore):
    
    s0, s1 = comp.shape
    if filter_dict['zero_pad']:
        s0fft = utils.nearest_product(s0,[2,3,5],'above')
        s1fft = utils.nearest_product(s1,[2,3,5],'above')
    else:
        s0fft, s1fft = s0, s1

    fake_map = np.empty([s0fft,s1fft])
    alm = pyfftw.empty_aligned(fake_map.shape, dtype='complex128')
    fft = pyfftw.builders.fft2(comp*apo,planner_effort='FFTW_ESTIMATE',threads=ncore,s=[s0fft,s1fft])
    alm = fft()
    if filter_dict['unpixwin']:
        wy, wx = calc_window(fake_map.shape)
        alm /= wy[:,None]
        alm /= wx[None,:]
    
    kfilter_x = gen_kx_mask(fake_map,filter_dict['d_th'],filter_dict['kx_cut'],filter_dict['kx_cut_apo'])
    kfilter_y = gen_ky_mask(fake_map,filter_dict['d_th'],filter_dict['ky_cut'],filter_dict['ky_cut_apo'])
    alm_filt = np.fft.ifftshift(alm)*kfilter_x*kfilter_y
    fft = pyfftw.builders.ifft2(np.fft.fftshift(alm_filt), planner_effort='FFTW_ESTIMATE',threads=ncore)
    ret = np.real(fft())[:s0,:s1]
    nonzero = np.where(apo!=0)
    zeros = np.where(apo==0)
    ret[nonzero] /= apo[nonzero]
    ret[zeros] = 0

    return ret

