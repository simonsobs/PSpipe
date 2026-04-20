"""
script to get theta_min, theta_max from SPT
"""
import sys

import numpy as np
import healpy as hp
from pspy import so_dict


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

release_dir = d["release_dir"]
nside= 8192
m = hp.read_map(f"{release_dir}/real_data_maps/full/full_095ghz.fits")
idx = np.where(m != hp.pixelfunc.UNSEEN)[0]
theta, _ = hp.pix2ang(nside, idx)
theta_range = (np.min(theta), np.max(theta))

print(theta_range)
