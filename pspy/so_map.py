"""
@brief: so map class for handling healpix and car maps
@author: this is  a wrapper around healpix and enlib (pixell).
"""

from __future__ import print_function
from pixell import enmap,reproject,enplot,curvedsky,powspec
from sph_tools import map2alm,alm2map
from pspy_utils import ps_lensed_theory_to_dict
import healpy as hp, pylab as plt, numpy as np, astropy.io.fits as pyfits
import sys,os,copy

class so_map:
    """
    Class describing a so map object.
    """
    def __init__(self):
        pass
    
    def copy(self):
        """
        @brief Create a copy of the so map object.
        """
        return copy.deepcopy(self)
    
    def info(self,showHeader=False):
        """
        @brief Print information about the so map object.
        """
        print ('pixellisation: ',self.pixel)
        print ('number of components: ',self.ncomp)
        if self.ncomp==1:
            print ('number of pixels: ', self.data.shape[:])
        else:
            print ('number of pixels: ', self.data.shape[1:])
        print ('nside: ', self.nside)
        print ('geometry:',self.geometry)
        print ('coordinates:', self.coordinate)
    
    def write_map(self,file_name):
        """
        @brief Write the so map.
        """
        if self.pixel=='HEALPIX':
            hp.fitsfunc.write_map(file_name, self.data,overwrite=True)
        if self.pixel=='CAR':
            enmap.write_map(file_name, self.data)

    def upgrade(self,factor):
        """
        @bried upgrade the so map
        @param factor need to be a factor of 2
        @return a so_map instance upgraded by factor.
        """
        assert( factor % 2 == 0), 'factor should be a factor of 2'
        
        upgrade=self.copy()
        if self.pixel=='HEALPIX':
            nside_out=nside*factor
            upgrade.data=hp.pixelfunc.ud_grade(self.data, nside_out=nside_out)
            upgrade.nside=nside_out
        if self.pixel=='CAR':
            upgrade.data=enmap.upgrade(self.data,factor)
            upgrade.geometry=upgrade.data.geometry[1:]
        return upgrade

    def downgrade(self,factor):
        """
        @brief downgrade the so map
        @param factor need to be a factor of 2
        @return a so_map instance downgraded by factor.
        """
        assert( factor % 2 == 0), 'factor should be a factor of 2'
        
        downgrade=self.copy()
        
        if self.pixel=='HEALPIX':
            nside_out=nside/factor
            downgrade.data=hp.pixelfunc.ud_grade(self.data, nside_out=nside_out)
            downgrade.nside=nside_out
        if self.pixel=='CAR':
            downgrade.data=enmap.downgrade(self.data,factor)
            downgrade.geometry=downgrade.data.geometry[1:]
        return downgrade
    
    def synfast(self,clfile):
        """
        @brief generate a cmb gaussian simulation in so map
        @param clfile: a lensed power spectrum file from CAMB
        @return: the so map with lensed CMB
        """
        if self.pixel=='HEALPIX':
            l,ps=ps_lensed_theory_to_dict(clfile,output_type='Cl',lstart=0)
            if self.ncomp==1:
                self.data= hp.sphtfunc.synfast(ps['TT'], self.nside ,new=True, verbose=False)
            else :
                self.data= hp.sphtfunc.synfast((ps['TT'],ps['EE'],ps['BB'],ps['TE']), self.nside ,new=True, verbose=False)

        if self.pixel=='CAR':
            ps=powspec.read_spectrum(clfile)[:self.ncomp,:self.ncomp]
            self.data= curvedsky.rand_map(self.data.shape, self.data.wcs, ps)

        return self

    def plot(self,color='planck',color_range=None,file_name=None,ticks_spacing_car=1,title='',cbar=True,hp_gnomv=None):
        """
        @brief Plot a so map, color is a maplotlib colormap or the planck colormap.
        @param color: a colormap
        @param color_range: should be a scalar if you want to plot only a single component, and a len(3) list if you want to plot T,Q,U
        @param file_name:  file_name is  the name of the png file that will be created, if None the plot will be displayed
        @param title: the title of the plot
        @param cbar: wether you display the colorbar or not
        @param ticks_spacing_CAR: for CAR plot, choose the spacing of the ticks
        @param hp_gnomv:  gnomview projection for HEALPIX plotting, expected (lon_c,lat_c,xsize,reso)
        """
        if self.pixel=='HEALPIX':
            if color=='planck':
                from matplotlib.colors import ListedColormap
                path=(os.path.dirname(os.path.realpath(__file__)))
                colombi1_cmap = ListedColormap(np.loadtxt('%s/../Planck_Parchment_RGB.txt'%path)/255.)
                colombi1_cmap.set_bad("white")
                colombi1_cmap.set_under("white")
                cmap = colombi1_cmap
            else:
                cmap= plt.get_cmap(color)
                cmap.set_bad("white")
                cmap.set_under("white")
            
            if self.ncomp==1:
                
                min,max=None,None
                if color_range is not None:
                    min=-color_range
                    max=color_range
            
                if hp_gnomv is not None:
                    lon,lat,xsize,reso=hp_gnomv
                    hp.gnomview(self.data,min=min,max=max,cmap=cmap, notext=True,title=title,cbar=cbar,rot=(lon,lat,0),xsize=xsize,reso=reso)
                else:
                    hp.mollview(self.data,min=min,max=max,cmap=cmap, notext=True,title=title,cbar=cbar)
                if file_name is not None:
                    plt.savefig(file_name+'.png', bbox_inches='tight')
                    plt.clf()
                    plt.close
                else:
                    plt.show()
            else:
                fields=['T','Q','U']
                min,max={},{}
                for l1 in fields:
                    min[l1],max[l1]=None,None
                if color_range is not None:
                    for i,l1 in enumerate(fields):
                        min[l1]=-color_range[i]
                        max[l1]=color_range[i]
                for map,l1 in zip(self.data,fields):
                    
                    if hp_gnomv is not None:
                        lon,lat,xsize,reso=hp_gnomv
                        hp.gnomview(self.data,min=min,max=max,cmap=cmap, notext=True,title=title,cbar=cbar,rot=(lon,lat,0),xsize=xsize,reso=reso)
                    else:
                        hp.mollview(map,min=min[l1],max=max[l1],cmap=cmap, notext=True,title=l1+''+title,cbar=cbar)
                    if file_name is not None:
                        plt.savefig(file_name+'_%s'%l1+'.png', bbox_inches='tight')
                        plt.clf()
                        plt.close
                    else:
                        plt.show()
                            
        if self.pixel=='CAR':
            if self.ncomp==1:
                if color_range is not None:
                    max='%s'%(color_range)
                else:
                    max='%s'%(np.max(self.data))
                
                plots = enplot.get_plots(self.data,color=color,range=max,colorbar=1,ticks=ticks_spacing_car)
                

                for plot in plots:
                    if file_name is not None:
                        enplot.write(file_name+'.png', plot)
                    else:
                        plot.img.show()

            if self.ncomp==3:
                fields=['T','Q','U']
    
                if color_range is not None:
                    max='%s:%s:%s'%(color_range[0],color_range[1],color_range[2])
                else:
                    max='%s:%s:%s'%(np.max(self.data[0]) ,np.max(self.data[1]),np.max(self.data[2]))

                plots = enplot.get_plots(self.data,color=color,range=max,colorbar=1,ticks=ticks_spacing_car)
    
                for (plot,l1) in zip(plots,fields):
                    if file_name is not None:
                        enplot.write(file_name+'_%s'%l1+'.png', plot)
                    else:
                        #enplot.show(plot,method="ipython")
                        plot.img.show()

def read_map(file,coordinate=None,verbose=False):
    """
    @brief Reads a FITS file and creates a so map object out of it.
    The FITS file can be either an enmap object or a healpix object.
    @return a so_map instance.
    """
    map = so_map()
    hdulist = pyfits.open(file)
    try:
        header = hdulist[1].header
        map.pixel='HEALPIX'
        map.ncomp= header['TFIELDS']
        map.data= hp.fitsfunc.read_map(file,field=np.arange(map.ncomp),verbose=False)
        map.nside=hp.pixelfunc.get_nside(map.data)
        map.geometry='healpix geometry'
        try:
            map.coordinate= header['SKYCOORD']
        except:
            map.coordinate=None

    except:
        header = hdulist[0].header
        map.pixel=(header['CTYPE1'][-3:])
        try:
            map.ncomp= header['NAXIS3']
        except:
            map.ncomp= 1
        map.data= enmap.read_map(file)
        map.nside=None
        map.geometry=map.data.geometry[1:]
        map.coordinate=header['RADESYS']
        if map.coordinate=='ICRS':
            map.coordinate='equ'

    if coordinate is not None:
        map.coordinate=coordinate

    return map

def healpix2car(map,template,lmax=None):
    """
    @brief Project a HEALPIX so map into a CAR so map
    the projection will be done in harmonic space, you can specify a lmax
    to choose a range of multipoles considered in the projection.
    If the coordinate of the map and the template differ, a rotation will be performed
    @param template: the car template you want to project into
    @param lmax: the maximum multipole in the HEALPIX map to project
    @return the projected map in the template pixellisation and coordinates
    """
    project=template.copy()
            
    if map.coordinate is None or template.coordinate is None:
        rot=None
    elif map.coordinate == template.coordinate:
        rot=None
    else:
        print ('will rotate from %s to %s coordinate system'%(map.coordinate,template.coordinate))
        rot="%s,%s"%(map.coordinate,template.coordinate)
    if lmax> 3*map.nside-1:
        print ('WARNING: your lmax is too large, setting it to 3*nside-1 now')
        lmax=3*map.nside-1
    if lmax is None:
        lmax=3*map.nside-1
    project.data=reproject.enmap_from_healpix(map.data, template.data.shape, template.data.wcs, ncomp=map.ncomp, unit=1, lmax=lmax,rot=rot, first=0)

    return project

def car2car(map,template):
    """
    @brief project a CAR map into another CAR map, see the pixell enmap.project documentation
    """
    project=template.copy()
    project.data=enmap.project(map.data,template.data.shape,template.data.wcs)
    return project

def healpix_template(ncomp,nside,coordinate=None):
    """
    @brief create a so map template with healpix pixellisation.
    @param ncomp: the number of component of the map can be 1 or 3
    @param nside
    @param coordinate: the coordinate of the template can be gal or equ
    """
    temp = so_map()
    
    if ncomp==3:
        temp.data=np.zeros((3,12*nside**2))
    else:
        temp.data=np.zeros((12*nside**2))

    temp.pixel='HEALPIX'
    temp.ncomp= ncomp
    temp.nside=nside
    temp.geometry='healpix geometry'
    temp.coordinate=coordinate
    return temp

def car_template(ncomp,ra0,ra1,dec0,dec1,res):
    """
    @brief create a so map template with car pixellisation in equ coordinates.
    @param ncomp: the number of component of the map can be 1 or 3
    @param ra0,dec0,ra1,dec1: in degrees
    @param res: resolution in arcminute
    @return: a so template with car pixellisation
    """
    if ncomp==3:
        pre=(3,)
    else:
        pre=()
    
    box=getbox(ra0,ra1,dec0,dec1)
    res=res*np.pi/(180*60)
    temp=so_map()
    shape,wcs= enmap.geometry(box, res=res,pre=pre)
    temp.data= enmap.zeros(shape, wcs=wcs, dtype=None)
    temp.pixel='CAR'
    temp.nside=None
    temp.ncomp=ncomp
    temp.geometry=temp.data.geometry[1:]
    temp.coordinate='equ'
    return temp

def getbox(ra0,ra1,dec0,dec1):
    """
    @brief create box in equatorial coordinate
    @param  ra0,dec0,ra1,dec1 in degrees
    """
    box= np.array( [[ dec0, ra1], [dec1, ra0]])*np.pi/180
    return(box)



                     


