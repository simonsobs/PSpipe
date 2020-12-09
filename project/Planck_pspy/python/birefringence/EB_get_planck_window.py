"""
This script is used to create window function for the EB analysis
(not in used if we use directly Minami&Komatsu window)
"""

import healpy as hp
from astropy.io import fits
import numpy as np
from pspy import so_window, so_map, so_dict
import sys


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# You have to spefify the data directory in which the products will be downloaded
data_dir = d["data_dir"]
freqs = d["freqs"]

maps_dir = data_dir + "/maps"
EB_mask_dir = data_dir + "/EB_masks"


nside = 2048
window = so_map.healpix_template(ncomp=1, nside=nside)

with fits.open("%s/HFI_Mask_PointSrc_2048_R2.00.fits" % EB_mask_dir) as hdul:
    data = hdul["SRC-POL"].data
ps_mask = {f: hp.reorder(data[f], n2r=True) for f in ["F100", "F143", "F217", "F353"]}


for freq in freqs:

    CO_mask = np.ones(12 * nside**2)
    
    if freq is not "143":
        log10_CO_noise_ratio = so_map.read_map("%s/HFI_BiasMap_%s-CO-noiseRatio_2048_R3.00_full.fits" % (EB_mask_dir, freq), fields_healpix = 0)
        id = np.where(log10_CO_noise_ratio.data > -2)
        CO_mask[id] = 0

    missing_pixel = np.ones(12*nside**2)
    half_mission = [1, 2]
    for hm in half_mission:
        for c, field in enumerate(["I", "Q", "U"]):
            map = so_map.read_map("%s/HFI_SkyMap_%s_2048_R3.01_halfmission-%s.fits" % (maps_dir, freq, hm), fields_healpix = c)
            id = np.where(map.data < -10**30)
            missing_pixel[id] = 0
        cov = so_map.read_map("%s/HFI_SkyMap_%s_2048_R3.01_halfmission-%s.fits" % (maps_dir, freq, hm), fields_healpix = 4)
        id = np.where(cov.data < 0)
        missing_pixel[id] = 0

    window.data[:] = missing_pixel * ps_mask["F%s" % freq] * CO_mask
    window = so_window.create_apodization(window, apo_type="C1", apo_radius_degree = 2)
    window.write_map("%s/window_%s.fits" % (EB_mask_dir, freq))
    window.plot(file_name="%s/window_%s" % (EB_mask_dir, freq))


