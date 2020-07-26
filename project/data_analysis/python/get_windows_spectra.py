"""
This script compute all power spectra of squared windows, it's a necessary step of covariance computation.
"""
from pspy import so_dict, so_map, sph_tools, so_spectra, pspy_utils, so_mpi
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
win_spec_dir = "win_spectra"

pspy_utils.create_directory(win_spec_dir)

na_list, nb_list, nc_list, nd_list, spec_name_list = [], [], [], [], []
n_alms = 0
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                # This ensures that we do not repeat redundant computations
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                na_list += [sv1]
                nb_list += [ar1]
                nc_list += [sv2]
                nd_list += [ar2]
                n_alms += 1
                
                spec_name = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                spec_name_list += [spec_name]

print("number of alms to compute : %s" % n_alms)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_alms - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = na_list[task], nb_list[task], nc_list[task], nd_list[task]
    spec_name = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)

    win_T1 = so_map.read_map(d["window_T_%s_%s" % (sv1, ar1)])
    win_T2 = so_map.read_map(d["window_T_%s_%s" % (sv2, ar2)])

    sq_win = win_T1.copy()
    sq_win.data[:] *= win_T2.data[:]
    alm_sqwin = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
    np.save("%s/alms_%s.npy" % (win_spec_dir, spec_name), alm_sqwin)

for spec_name1 in spec_name_list:
    for spec_name2 in spec_name_list:
        alm1 = np.load("%s/alms_%s.npy" % (win_spec_dir, spec_name1))
        alm2 = np.load("%s/alms_%s.npy" % (win_spec_dir, spec_name2))

        l, wcl = so_spectra.get_spectra_pixell(alm1, alm2)
        np.savetxt("%s/win_spectrum_%s_%s.dat" % (win_spec_dir, spec_name1, spec_name2), np.transpose([l, wcl]))
