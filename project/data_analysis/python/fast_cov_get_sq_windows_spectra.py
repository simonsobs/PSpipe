"""
This script compute all power spectra of squared window alms, it's a necessary step of covariance computation.
"""
from pspy import so_dict, so_map, sph_tools, so_spectra, pspy_utils, so_mpi
import numpy as np
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
win_spec_dir = "win_spectra"
pspy_utils.create_directory(win_spec_dir)

spec_name_list = []
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                # This ensures that we do not repeat redundant computations
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue

                spec_name = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                spec_name_list += [spec_name]

for sid1, spec_name1 in enumerate(spec_name_list):
    for sid2, spec_name2 in enumerate(spec_name_list):
        if sid1 > sid2: continue
        print(spec_name1, spec_name2)
        alm1 = np.load("%s/alms_%s.npy" % (win_spec_dir, spec_name1))
        alm2 = np.load("%s/alms_%s.npy" % (win_spec_dir, spec_name2))
        l, wcl = so_spectra.get_spectra_pixell(alm1, alm2)
        np.savetxt("%s/win_spectrum_%s_%s.dat" % (win_spec_dir, spec_name1, spec_name2), np.transpose([l, wcl]))

os.system("rm %s/alm*" % win_spec_dir)
