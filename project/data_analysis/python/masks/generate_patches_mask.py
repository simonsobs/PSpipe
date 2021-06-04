import os
import sys

import numpy as np
import pandas as pd
from pixell import enmap
from pspy import so_dict, so_map

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

binary = so_map.read_map(d["template"])
if binary.data.ndim > 2:
    # Only use temperature
    binary.data = binary.data[0]
binary.ncomp = 1
binary.data = binary.data.astype(np.int16)
binary.data[:] = 0

for i in range(d["nbr_patches"]):
    patch = f"patch_{i}"
    if patch not in d:
        raise ValueError(f"Missing '{patch}' coordinates")

    mask = binary.copy()

    # Grab pixels given coordinates
    dec_ra = np.deg2rad(d[patch])
    pix1 = enmap.sky2pix(mask.data.shape, mask.data.wcs, dec_ra[0])
    pix2 = enmap.sky2pix(mask.data.shape, mask.data.wcs, dec_ra[1])
    min_pix = np.min([pix1, pix2], axis=0).astype(int)
    max_pix = np.max([pix1, pix2], axis=0).astype(int)

    mask.data[min_pix[0] : max_pix[0], min_pix[1] : max_pix[1]] = 1

    patch_path = os.path.join(d.get("patch_output_dir", "."), patch)
    mask.downgrade(4).plot(file_name=patch_path)
    mask.write_map(f"{patch_path}.fits")
