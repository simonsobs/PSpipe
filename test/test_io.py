from pspy import so_map
from pspy import sph_tools
from pixell import curvedsky,powspec

"""
This is an example and how to read/write/plot so map
"""

ra0,ra1,dec0,dec1=-5,5,-5,5
res=1
ncomp=3
clfile='../data/bode_almost_wmap5_lmax_1e4_lensedCls.dat'


template_car= so_map.car_template(ncomp,ra0,ra1,dec0,dec1,res)
template_healpix= so_map.healpix_template(ncomp,nside=256,coordinate='equ')

cmb_car=template_car.synfast(clfile)
cmb_healpix=template_healpix.synfast(clfile)

cmb_car.plot()
cmb_healpix.plot()

cmb_car.write_map('map_car.fits')
cmb_healpix.write_map('map_healpix.fits')


cmb_car2=so_map.read_map('map_car.fits')
cmb_healpix2=so_map.read_map('map_healpix.fits')

cmb_car2.data-=cmb_car.data
cmb_healpix2.data-=cmb_healpix.data

cmb_car2.plot()
cmb_healpix2.plot()

