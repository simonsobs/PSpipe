"""
This script compute all alms squared windows, it's a necessary step of covariance computation.
"""
from pspy import so_dict, so_map, sph_tools, so_spectra, pspy_utils, so_mpi
from pspipe_utils import pspipe_list
import numpy as np
import sys


def mult(map_a, map_b):

    res_a = 1 / map_a.data.pixsize()
    res_b = 1 / map_b.data.pixsize()

    if res_a == res_b:
        prod = map_a.copy()
        prod.data *= map_b.data
    elif res_a < res_b:
        print("resample map a")
        prod = map_b.copy()
        map_a_proj = so_map.car2car(map_a, map_b)
        prod.data *= map_a_proj.data
    elif res_b < res_a:
        print("resample map b")
        prod = map_a.copy()
        map_b_proj = so_map.car2car(map_b, map_a)
        prod.data *= map_b_proj.data
    
    return prod


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
sq_win_alms_dir = "sq_win_alms"

pspy_utils.create_directory(sq_win_alms_dir)

n_sq_alms, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)


print("number of sq win alms to compute : %s" % n_sq_alms)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_sq_alms - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]

    win_T1 = so_map.read_map(d["window_T_%s_%s" % (sv1, ar1)])
    win_T2 = so_map.read_map(d["window_T_%s_%s" % (sv2, ar2)])

    sq_win = mult(win_T1, win_T2)
    #sq_win = win_T1.copy()
    #sq_win.data[:] *= win_T2.data[:]
    sqwin_alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)
    
    np.save("%s/alms_%s_%sx%s_%s.npy" % (sq_win_alms_dir, sv1, ar1, sv2, ar2), sqwin_alm)

