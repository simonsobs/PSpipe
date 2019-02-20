"""
This is an example on how to read/write/plot so map
"""
import matplotlib
matplotlib.use('Agg')
from pspy import so_map,sph_tools
from pixell import curvedsky,powspec
import os

test_dir='result_test_io'
try:
    os.makedirs(test_dir)
except:
    pass

#We start with the definition of the CAR template, it will go from ra0 to ra1 and from dec0 to dec1 (all in degrees)
#It will have a resolution of 1 arcminute
#It allow 3 components (stokes parameter in the case of CMB anisotropies)

ra0,ra1,dec0,dec1=-5,5,-5,5
res=1
ncomp=3
clfile='../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat'

#We generate both a CAR and HEALPIX templates

template_car= so_map.car_template(ncomp,ra0,ra1,dec0,dec1,res)
template_healpix= so_map.healpix_template(ncomp,nside=256,coordinate='equ')

#We fill the templates with CMB data

cmb_car=template_car.synfast(clfile)
cmb_healpix=template_healpix.synfast(clfile)

#We plot them

cmb_car.plot(file_name='%s/map_car_io_test'%test_dir)
cmb_healpix.plot(file_name='%s/map_healpix_io_test'%test_dir)

#We write them to disk

cmb_car.write_map('%s/map_car.fits'%test_dir)
cmb_healpix.write_map('%s/map_healpix.fits'%test_dir)

#We read the maps

cmb_car2=so_map.read_map('%s/map_car.fits'%test_dir)
cmb_healpix2=so_map.read_map('%s/map_healpix.fits'%test_dir)

#We null them

cmb_car2.data-=cmb_car.data
cmb_healpix2.data-=cmb_healpix.data

#We plot the nulls: note that while car is zero, some low amplitude numerical noise is there in healpix

cmb_car2.plot(file_name='%s/map_car_null'%test_dir)
cmb_healpix2.plot(file_name='%s/map_healpix_null'%test_dir)

