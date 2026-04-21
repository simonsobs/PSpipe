"""
This script coadd the 30 splits of SPT into 6 independent splits
"""

import sys

import numpy as np
import healpy as hp
from pspy import so_map, pspy_utils


release_dir = "/global/cfs/cdirs/sobs/users/tlouis/spt_data/"
arrays_spt = ["095", "150", "220"]

out_dir = release_dir + "real_data_maps/one_sixth/"
pspy_utils.create_directory(out_dir)


n_split_in = 30
id_split = np.arange(n_split_in)
n_split_out = 6
n_group = n_split_in // n_split_out
nsplit_per_group = n_split_in // n_group

groups = np.split(id_split, n_split_out)

print("n_group", n_group)
print("nsplit_per_group", nsplit_per_group)
print("groups", groups)
print("")

template = so_map.read_map(release_dir + "real_data_maps/one_thirtieth/one_thirtieth_bundle00_095ghz.fits")

for g in groups:

    for ar in arrays_spt:
    
        template.data[:] = 0
        
        for iii in g:
            split = so_map.read_map(release_dir + f"real_data_maps/one_thirtieth/one_thirtieth_bundle{iii:02d}_{ar}ghz.fits")
            template.data[:] += split.data[:]
            print(iii, ar)
            
        print("")
        template.data[:] /= nsplit_per_group
        hp.write_map(f"{out_dir}/one_sixth_bundle{iii:02d}_{ar}ghz.fits", template.data[:], overwrite=True, partial=True)
