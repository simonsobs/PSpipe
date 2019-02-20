"""
This is a test of generation of simulations and projection.
We first specify two templates, one in equ coordnates and CAR pixellisation and one in equ coordinates and HEALPIX pixellisation.
We generate alms from a CAMB lensed power spectrum file and use them to generate a random CMB realisation in both template.
We then project the HEALPIX simulation and plot both the native CAR simulation and the projected HEALPIX simulation.
We chose a low resolution nside to emphasize the effect of resolution
"""
import matplotlib
matplotlib.use('Agg')
from pspy import so_map,sph_tools
from pixell import curvedsky,powspec
import os

test_dir='result_test_projection'
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
lmax=5000

#We read a CAMB lensed power spectrum and generate alms from it
clfile='../data/bode_almost_wmap5_lmax_1e4_lensedCls.dat'
ps=powspec.read_spectrum(clfile)[:ncomp,:ncomp]

alms= curvedsky.rand_alm(ps, lmax=lmax)

#We generate both a CAR and HEALPIX template
#We choose nside=256 so that the resolution of HEALPIX is much smaller
template_car= so_map.car_template(ncomp,ra0,ra1,dec0,dec1,res)
template_healpix= so_map.healpix_template(ncomp,nside=256,coordinate='equ')

#We compule the stokes parameters from the alms in both templates
map_healpix=sph_tools.alm2map(alms,template_healpix)
map_car=sph_tools.alm2map(alms,template_car)

#We project the healpix map into the CAR template
map_healpix_projected= so_map.healpix2car(map_healpix,map_car,lmax=lmax)

#We plot both the native CAR maps and the Healpix projected to CAR maps
#They contain the same CMB but have different resolutions.
map_car.plot(file_name='%s/map_car'%test_dir)
map_healpix_projected.plot(file_name='%s/map_healpix_projected'%test_dir)


