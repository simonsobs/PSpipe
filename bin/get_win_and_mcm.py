#!/usr/bin/env python

from __future__ import print_function
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

result_dir=d['result_dir']
pspy_utils.create_directory(result_dir)

if d['spin']=='0-2':
    ncomp=3
    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']
elif d['spin']=='0':
    ncomp=1

if d['pixel']=='CAR':
    print (d['ra0'],d['ra1'],d['dec0'],d['dec1'],d['res'])
    binary=so_map.car_template(1,d['ra0'],d['ra1'],d['dec0'],d['dec1'],d['res'])
    binary.data[:]=0
    binary.data[1:-1,1:-1]=1

elif d['pixel']=='HEALPIX':
    binary=so_map.healpix_template(ncomp=1,nside=d['nside'])
    vec=hp.pixelfunc.ang2vec(d['lon'],d['lat'], lonlat=True)
    disc=hp.query_disc(d['nside'], vec, radius=d['radius']*np.pi/180)
    binary.data[disc]=1


window=so_window.create_apodization(binary, apo_type=d['apo_type_survey'], apo_radius_degree=d['apo_radius_survey'])
mask=so_map.simulate_source_mask(binary, nholes=d['source_mask_nholes'], hole_radius_arcmin=d['source_mask_radius'])
mask= so_window.create_apodization(mask, apo_type=d['apo_type_mask'], apo_radius_degree=d['apo_radius_mask'])
window.data*=mask.data

window.write_map('%s/window.fits'%(result_dir))

if ncomp==3:
    window=(window,window)
    mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2(window, d['binning_file'], lmax=d['lmax'], type=d['type'],save_file='%s/test'%result_dir)
if ncomp==1:
    mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0(window, d['binning_file'], lmax=d['lmax'], type=d['type'],save_file='%s/test'%result_dir)


