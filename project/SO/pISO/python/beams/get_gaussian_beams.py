import numpy as np
import healpy as hp

from pspy import so_dict

import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

beam_dir = d['beam_dir_lat_iso']

l = np.arange(12000)

fwhms = [7.4, 5.1, 2.2, 1.5, 0.96, 0.8]

for f in fwhms:
    bl = hp.gauss_beam(np.deg2rad(f/60), lmax=l[-1])
    beam = np.zeros((l.size, 3))
    beam[:, 0] = l
    beam[:, 1] = bl
    beam[:, 2:] = 0 # no error
    np.savetxt(beam_dir + f'beam_gaussian_fwhm_{f}_arcmin_no_error.txt', beam)