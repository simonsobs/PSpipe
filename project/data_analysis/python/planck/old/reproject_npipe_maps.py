import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from pixell import sharp,curvedsky
from scipy.interpolate import interp1d
from pixell import reproject
import os
from soapack import interfaces as sints

def removeDipole(m,dipole,monopole=None,copy=True, bad=hp.UNSEEN,nest=False):
    if copy:
        m = m.copy()
    npix = m.size
    nside = hp.npix2nside(npix)
    if nside > 128:
        bunchsize = npix // 24
    else:
        bunchsize = npix
#     mono, dipole = fit_dipole(m, nest=nest, bad=bad, gal_cut=gal_cut)
    for ibunch in range(npix // bunchsize):
        ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
        ipix = ipix[(m.flat[ipix] != bad) & (np.isfinite(m.flat[ipix]))]
        x, y, z = hp.pix2vec(nside, ipix, nest)
        m.flat[ipix] -= dipole[0] * x
        m.flat[ipix] -= dipole[1] * y
        m.flat[ipix] -= dipole[2] * z
        if monopole is not None:
            m.flat[ipix] -= monopole
    return m

def map2alm(hp_map,inversePolTransferFunc,ncomp=3,dtype = np.float64, first=0, unit=1e-6,lmax=0): #Unit  convert to uk
    assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"
    ctype = np.result_type(dtype, 0j)
    # Read the input maps
    if type(hp_map) == str:
        m = np.atleast_2d(hp.read_map(hp_map, field=tuple(
            range(first, first + ncomp)))).astype(dtype)
    else:
        m = np.atleast_2d(hp_map).astype(dtype)
    if unit != 1:
        m /= unit
    # Prepare the transformation
    print("Preparing SHT")
    nside = hp.npix2nside(m.shape[1])
    lmax = lmax or 3 * nside
    minfo = sharp.map_info_healpix(nside)
    ainfo = sharp.alm_info(lmax)
    sht = sharp.sht(minfo, ainfo)
    alm = np.zeros((ncomp, ainfo.nelem), dtype=ctype)
    # Perform the actual transform
    print("T -> alm")
    print(m.dtype, alm.dtype)
    sht.map2alm(m[0], alm[0])
    if ncomp == 3:
        print("P -> alm")
        sht.map2alm(m[1:3], alm[1:3], spin=2)
        if inversePolTransferFunc is not None:
            alm[1] = curvedsky.almxfl(alm[1],inversePolTransferFunc)
            alm[2] = curvedsky.almxfl(alm[2],inversePolTransferFunc)
    del m
    return alm

# From NPIPE paper
amp = 3366.6/1e6 #k
l = 263.986
b = 48.247

vec = hp.pixelfunc.ang2vec((90-b)/180*np.pi,l/180.*np.pi)
vec*= amp
est_monopole = {30:773/1e6,44:1187/1e6,70:70.8/1e6,100:-70./1e6,143:-81./1e6,217:-182.4/1e6,353:395.2/1e6,545:-5003/1e6,857:-.72}

# # Tabulated transfer funcs from paper. I found the files on NERSC so can use them if I reproject again. 
# # Whilst I used the tabulated versions for the current reprojection. It is commented out as the other method is preferable in future.
# transferFuncPath = '/global/homes/w/wcoulton/ACT/data/npipe_60pc_Emode_trans.dat'
# transferFunc = np.loadtxt(transferFuncPath,unpack=True)
# transferFunc_ell = transferFunc[0].astype('int').copy()
# index = {30:1,44:3,70:5,100:7,143:9,217:11,353:13}

inputDir = '/global/cfs/cdirs/cmb/data/planck2020/npipe/'
outputDir = '/global/cscratch1/sd/alaposta/npipe6v20_projected_will/'

for freq in [353][::-1]:
    for split in ['A','B']:
        print(f'Doing {freq} and split {split}')
        shape,wcs = sints.get_geometry('p01')
        oMapName = f'{outputDir}/npipe6v20{split}_{freq:03d}_map_enmap_monoSub.fits'
        if not os.path.exists(oMapName):
            if freq in [545,857]:
                maps = hp.read_map(f'{inputDir}/npipe6v20{split}/npipe6v20{split}_{freq:03d}_map.fits',field=[0])
                maps = removeDipole(maps,vec,monopole=est_monopole[freq])

                # fl = pyfits.open(f'{inputDir}/npipe6v20{split}/quickpol/Bl_TEB_npipe6v20_{freq:03d}{split}x{freq:03d}{split}_only_E_tf_60fsky.fits')
                # transferFunc = fl[1].data['E']
                # invtransferFunc = 1/transferFunc
                # invtransferFunc[transferFunc==0] = 0
                # inversePolTransferFunc = interp1d(np.arange(transferFunc.shape[0]),invtransferFunc)

                # if freq in index.keys():
                #     temp = np.ones(100000)
                #     temp[transferFunc_ell] = 1/transferFunc[index[freq]]
                #     tempFnc = interp1d(np.arange(100000),temp)
                #     inversePolTransferFunc = lambda x: tempFnc(x)
                # else:
                #     inversePolTransferFunc = lambda x: x*0+1.
                alms = map2alm(maps,None,ncomp=1)

                
                maps_enmap = reproject.enmap_from_healpix(alms, shape, wcs, ncomp=1,is_alm = True)
            else:
                maps = hp.read_map(f'{inputDir}/npipe6v20{split}/npipe6v20{split}_{freq:03d}_map.fits',field=[0,1,2])
                maps[0] = removeDipole(maps[0],vec,monopole=est_monopole[freq])

                fl = pyfits.open(f'{inputDir}/npipe6v20{split}/quickpol/Bl_TEB_npipe6v20_{freq:03d}{split}x{freq:03d}{split}_only_E_tf_60fsky.fits')
                transferFunc = fl[1].data['E']
                invtransferFunc = 1/transferFunc
                invtransferFunc[transferFunc==0] = 0
                inversePolTransferFunc = interp1d(np.arange(transferFunc.shape[0]),invtransferFunc)

                # if freq in index.keys():
                #     temp = np.ones(100000)
                #     temp[transferFunc_ell] = 1/transferFunc[index[freq]]
                #     tempFnc = interp1d(np.arange(100000),temp)
                #     inversePolTransferFunc = lambda x: tempFnc(x)
                # else:
                #     inversePolTransferFunc = lambda x: x*0+1.
                alms = map2alm(maps,inversePolTransferFunc)

                
                maps_enmap = reproject.enmap_from_healpix(alms, shape, wcs, ncomp=3,is_alm = True)
            maps_enmap.write(oMapName)
        oVarName= f'{outputDir}/npipe6v20{split}_{freq:03d}_ivar_enmap.fits'
        if not os.path.exists(oVarName):
            if freq in [545,857]:
                var = hp.read_map(f'{inputDir}/npipe6v20{split}/npipe6v20{split}_{freq:03d}_wcov_hrscaled.fits',field=[0])
            else:
                var = hp.read_map(f'{inputDir}/npipe6v20{split}/npipe6v20{split}_{freq:03d}_wcov_mcscaled.fits',field=[0])
            var[var!=0]= 1/(var*1e12) #Convert to uk

            ivar_enmap = reproject.enmap_from_healpix_interp(var, shape, wcs)#, ncomp=1)

            ivar_enmap.write(oVarName)

        oHitsName= f'{outputDir}/npipe6v20{split}_{freq:03d}_hmap_enmap.fits'
        if not os.path.exists(oHitsName):
            hits = hp.read_map(f'{inputDir}/npipe6v20{split}/npipe6v20{split}_{freq:03d}_hmap.fits',field=[0])
            
            hits_enmap = reproject.enmap_from_healpix_interp(hits, shape, wcs)#, ncomp=1)

            hits_enmap.write(oHitsName)


shape,wcs = sints.get_geometry('p01')
"""
TTYPE2  = 'GAL040  '           / 40% sky coverage
TTYPE3  = 'GAL060  '           / 60% sky coverage
TTYPE4  = 'GAL070  '           / 70% sky coverage
TTYPE5  = 'GAL080  '           / 80% sky coverage
TTYPE6  = 'GAL090  '           / 90% sky coverage
TTYPE7  = 'GAL097  '           / 97% sky coverage
TTYPE8  = 'GAL099  '           / 99% sky coverage
"""
mask = hp.read_map(f'{outputDir}/masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=[3])
mask_enmap = reproject.enmap_from_healpix_interp(mask, shape, wcs)
mask_enmap.write(f'/{outputDir}/masks/galMask_80pc.fits')
mask = hp.read_map(f'/{outputDir}/masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=[2])
mask_enmap = reproject.enmap_from_healpix_interp(mask, shape, wcs)
mask_enmap.write(f'{outputDir}/masks/galMask_70pc.fits')
mask = hp.read_map(f'{outputDir}/masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=[1])
mask_enmap = reproject.enmap_from_healpix_interp(mask, shape, wcs)
mask_enmap.write(f'{outputDir}/masks/galMask_60pc.fits')