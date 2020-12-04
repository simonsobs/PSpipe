"""
This script is used to download complementary data for the EB analysis
"""

import numpy as np
from pspy import pspy_utils, so_dict
import sys
import wget
import tarfile
import astropy.io.fits as fits

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# You have to spefify the data directory in which the products will be downloaded
data_dir = d["data_dir"]
freqs = d["freqs"]

pspy_utils.create_directory(data_dir)

EB_mask_dir = data_dir + "/EB_masks"
pspy_utils.create_directory(EB_mask_dir)
# Planck keep inconsistent notation for the halfmission, sometimes 'halfmission-1' sometimes 'hm1'
for f in freqs:
    if f == "143": continue
    url = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_BiasMap_%s-CO-noiseRatio_2048_R3.00_full.fits" % (f)
    print(url)
    wget.download(url, "%s/HFI_BiasMap_%s-CO-noiseRatio_2048_R3.00_full.fits" % (EB_mask_dir, f))

url = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_Mask_PointSrc_2048_R2.00.fits"
print(url)
wget.download(url, "%s/HFI_Mask_PointSrc_2048_R2.00.fits" % (EB_mask_dir))

