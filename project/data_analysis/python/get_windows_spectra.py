"""
This script compute all power spectra of squared windows, it's a necessary step of covariance computation.
"""
from pspy import so_dict, so_map, sph_tools, so_spectra, pspy_utils
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
win_spec_dir = "win_spectra"

pspy_utils.create_directory(win_spec_dir)

spec_name_list = []
alm_sqwin = {}
for sv1 in surveys:
    arrays_1 = d["arrays_%s" % sv1]
    for ar1 in arrays_1:
        win_T1 = so_map.read_map(d["window_T_%s_%s" % (sv1, ar1)])
        for sv2 in surveys:
            arrays_2 = d["arrays_%s" % sv2]
            for ar2 in arrays_2:
                win_T2 = so_map.read_map(d["window_T_%s_%s" % (sv2, ar2)])

                sq_win = win_T1.copy()
                sq_win.data[:] *= win_T2.data[:]
                
                spec_name = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                print(spec_name)
                
                alm_sqwin[spec_name] = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
                spec_name_list += [spec_name]


for spec_name1 in spec_name_list:
    for spec_name2 in spec_name_list:
    
        l, wcl = so_spectra.get_spectra_pixell(alm_sqwin[spec_name1], alm_sqwin[spec_name2])
        np.savetxt("%s/win_spectrum_%s_%s.dat" % (win_spec_dir, spec_name1, spec_name2), np.transpose([l, wcl]))
