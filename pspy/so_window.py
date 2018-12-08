"""
@brief: python routines for window function generation
"""
from __future__ import absolute_import, print_function
import healpy as hp, pylab as plt, numpy as np, astropy.io.fits as pyfits
from pixell import enmap
import scipy
import tempfile
import os, sys
import shutil

def get_distance(binary):
    """
    @brief get the distance to the closest masked pixels for CAR and healpix pixellisation.
    This routine is not ideal, for healpix we use the excecutable process mask from the healpix fortran distribution
    For CAR we use the scipy distance transform that assume that pixel are all of the same size.
    @param binary, a so_map with binary data (1 is observed, 0 is masked)
    @return the distance to the closest masked pixels in degree
    """
    
    def write_dict_file(tempdir):
        file = open("%s/distance.dict"%tempdir,'w')
        file.write("mask_file=%s/tempmask \n"%tempdir)
        file.write("hole_min_size=0 \n")
        file.write("hole_min_surf_arcmin2=0.0 \n")
        file.write("filled_file='' \n")
        file.write("distance_file=%s/tempfile \n"%tempdir)
        file.close()
        return
    
    dist=binary.copy()
    if binary.pixel=='HEALPIX':

        tempdir=tempfile.mkdtemp()
        hp.fitsfunc.write_map('%s/tempmask'%tempdir, binary.data)
        write_dict_file(tempdir)
        os.system('source $HOME/.profile; process_mask %s/distance.dict'%tempdir)
        dist.data=hp.fitsfunc.read_map('%s/tempfile'%tempdir)
        shutil.rmtree(tempdir)
        dist.data*=180/np.pi
    
    if binary.pixel=='CAR':
        pixSize_arcmin= np.sqrt(binary.data.pixsize()*(60*180/np.pi)**2)
        dist.data[:]= scipy.ndimage.distance_transform_edt(binary.data)
        dist.data[:]*=pixSize_arcmin/60

    return dist


def create_apodization(binary, apo_type, apo_radius_degree):
    """
    @brief create a apodized window from a binary mask.
    @param binary: a so map binary mask
    @param apo_type: the type of apodisation you want to use
    @param apo_radius: the radius of apodisation in degrees
    @return a apodized window function
    """

    if apo_type=='C1':
        window=apod_C1(binary,apo_radius_degree)
    if apo_type=='C2':
        window=apod_C2(binary,apo_radius_degree)
    if apo_type=='Rectangle':
        if binary.pixel=='HEALPIX':
            print( 'no rectangle apod for healpix map')
            sys.exit()
        if binary.pixel=='CAR':
            window= apod_rectange(binary,apo_radius_degree)

    return window

def apod_C2(binary,radius):
    """
    @brief C2 apodisation as defined in https://arxiv.org/pdf/0903.2350.pdf
    """
    
    if radius==0:
        return binary
    else:
        dist=get_distance(binary)
        win=binary.copy()
        id=np.where(dist.data> radius)
        win.data=dist.data/radius-np.sin(2*np.pi*dist.data/radius)/(2*np.pi)
        win.data[id]=1
        
    return(win)

def apod_C1(binary,radius):
    """
    @brief C1 apodisation as defined in https://arxiv.org/pdf/0903.2350.pdf
    """
    
    if radius==0:
        return binary
    else:
        dist=get_distance(binary)
        win=binary.copy()
        id=np.where(dist.data> radius)
        win.data=1./2-1./2*np.cos(-np.pi*dist.data/radius)
        win.data[id]=1
    
    return(win)

def apod_rectange(binary,radius):
    """
    @brief apodisation suitable for rectangle window (in CAR) (smoother at the corner)
    """
    
    if radius==0:
        return binary
    else:
        shape= binary.data.shape
        wcs= binary.data.wcs
        Ny,Nx=shape
        pixScaleY,pixScaleX= enmap.pixshape(shape, wcs)
        win=binary.copy()
        win.data= win.data*0+1
        winX=win.copy()
        winY=win.copy()
        Id=np.ones((Ny,Nx))
        degToPix_x=np.pi/180/pixScaleX
        degToPix_y=np.pi/180/pixScaleY
        lenApod_x=int(radius*degToPix_x)
        lenApod_y=int(radius*degToPix_y)
    
        for i in range(lenApod_x):
            r=float(i)
            winX.data[:,i]=1./2*(Id[:,i]-np.cos(-np.pi*r/lenApod_x))
            winX.data[:,Nx-i-1]=winX.data[:,i]
        for j in range(lenApod_y):
            r=float(j)
            winY.data[j,:]=1./2*(Id[j,:]-np.cos(-np.pi*r/lenApod_y))
            winY.data[Ny-j-1,:]=winY.data[j,:]

        win.data=winX.data*winY.data
        return(win)



