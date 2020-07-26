"""
This script compute all alms squared windows, it's a necessary step of covariance computation.
"""
from pspy import so_dict, so_map, sph_tools, so_spectra, pspy_utils, so_mpi
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
sq_win_alms_dir = "sq_win_alms"

pspy_utils.create_directory(win_spec_dir)

na_list, nb_list, nc_list, nd_list = [], [], [], []
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

print("number of sq win alms to compute : %s" % n_alms)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_alms - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = na_list[task], nb_list[task], nc_list[task], nd_list[task]

    win_T1 = so_map.read_map(d["window_T_%s_%s" % (sv1, ar1)])
    win_T2 = so_map.read_map(d["window_T_%s_%s" % (sv2, ar2)])

    sq_win = win_T1.copy()
    sq_win.data[:] *= win_T2.data[:]
    sqwin_alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
    
    np.save("%s/alms_%s_%sx%s_%s.npy" % (sq_win_alms_dir, sv1, ar1, sv2, ar2), sqwin_alm)

