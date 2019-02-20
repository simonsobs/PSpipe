"""
This is a test of generation of apodised window functions.
We first specify a CAR template (100 sq degree), we construct a binary mask by setting pixels at the edges to be 0 and the rest of the map to be 1
We then create window function with 'C1', 'C2' and a rectangle apodisation.
We then generate a Healpix template at nside =512, we set all pixels to zero apart from pixels in a disc at longitude 20 degree and latitude
30 degree, we then create window functions with 'C1' and 'C2'  apodisation
"""
import matplotlib
matplotlib.use('Agg')
from pspy import so_map,so_window
import healpy as hp, numpy as np
import os

#The radius of the window
apo_radius=1
#The number of component of the binary mask (should be one)
ncomp=1

test_dir='result_window_function'
try:
    os.makedirs(test_dir)
except:
    pass

# We start with the definition of a CAR template, it will go from ra0 to ra1 and from dec0 to dec1 (all in degrees)
# It will have a resolution of 1 arcminute
# It is made with 1 component, corresponding to the binary mask

ra0,ra1,dec0,dec1=-5,5,-5,5
res=1
binary_car=so_map.car_template(ncomp,ra0,ra1,dec0,dec1,res)

# We set all pixel to 1 but the one at the border (set to zero) thus emulating a binary mask

binary_car.data[:,:]=0.
binary_car.data[1:-1,1:-1]=1

# We use three types of apodisation:
# C1 and C2 based on the distance to the closest masked pixel.
# Rectangle a apodisation suitable for rectangle patch in CAR (smoother at the corner)

for apo_type in ['C1','C2','Rectangle']:
    win = so_window.create_apodization(binary_car, apo_type, apo_radius_degree=apo_radius)
    win.plot(file_name='%s/window_CAR_%s'%(test_dir,apo_type))

# We now do the same for a HEALPIX template, with nside= 512
# We emulate a binary mask by setting all the pixel to zero but the ones
# in a disc of radius 5degree at longitude lon and lattitude lat

nside=512
binary_healpix=so_map.healpix_template(ncomp,nside=nside)
lon,lat,xsize,reso=20,30,500,2
vec= hp.pixelfunc.ang2vec(lon,lat, lonlat=True)
disc=hp.query_disc(nside, vec, radius=5*np.pi/180)
binary_healpix.data[disc]=1

# We use two types of apodisation:
# C1 and C2 based on the distance to the closest masked pixel
# And show the result in gnomview projection

for apo_type in ['C1','C2']:
    win = so_window.create_apodization(binary_healpix, apo_type, apo_radius_degree=apo_radius)
    win.plot(hp_gnomv=(lon,lat,xsize,reso), file_name='%s/window_healpix_%s'%(test_dir,apo_type))


