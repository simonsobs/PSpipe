'''
Just copy the windows
'''
import numpy as np
import healpy as hp
from pspy import so_dict, so_map, so_mcm, pspy_utils
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"

pspy_utils.create_directory(windows_dir)

freqs = d["freqs"]
splits = d["splits"]
experiment = "Planck"

print("Compute Planck 2018 mode coupling matrices")

for f1, freq1 in enumerate(freqs):
    
    window_t_1 = d["window_T_%s" % freq1]
    window_pol_1 = d["window_pol_%s" % freq1]

    for count1, hm1 in enumerate(splits):
        
        win_t1 = so_map.read_map(window_t_1[count1])
        win_pol1 = so_map.read_map(window_pol_1[count1])
        
        win_t1.write_map("%s/window_T_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))
        win_pol1.write_map("%s/window_P_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))

