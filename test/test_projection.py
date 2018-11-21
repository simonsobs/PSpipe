from pspy import so_map
from pspy import sph_tools
from pixell import curvedsky,powspec



ra0,ra1,dec0,dec1=-5,5,-5,5
res=1
ncomp=3
lmax=3000

clfile='../data/bode_almost_wmap5_lmax_1e4_lensedCls.dat'
ps=powspec.read_spectrum(clfile)[:ncomp,:ncomp]
alms= curvedsky.rand_alm(ps)

template_car= so_map.car_template(ncomp,ra0,ra1,dec0,dec1,res)
template_healpix= so_map.healpix_template(ncomp,nside=256,coordinate='equ')

map_healpix=sph_tools.alm2map(alms,template_healpix)
map_car=sph_tools.alm2map(alms,template_car)
map_healpix_projected= so_map.healpix2car(map_healpix,map_car,lmax=None)

map_car.plot(file_name='map_car')
map_healpix_projected.plot(file_name='map_healpix_projected')


