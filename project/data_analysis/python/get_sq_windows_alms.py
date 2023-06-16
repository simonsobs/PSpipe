"""
This script compute all alms squared windows, it's a necessary step of covariance computation.
"""
import sys

import numpy as np
from pspipe_utils import log, pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, sph_tools


def mult(map_a, map_b, log):
    res_a = 1 / map_a.data.pixsize()
    res_b = 1 / map_b.data.pixsize()

    if res_a == res_b:
        prod = map_a.copy()
        prod.data *= map_b.data
    elif res_a < res_b:
        log.info("resample map a")
        prod = map_b.copy()
        map_a_proj = so_map.car2car(map_a, map_b)
        prod.data *= map_a_proj.data
    elif res_b < res_a:
        log.info("resample map b")
        prod = map_a.copy()
        map_b_proj = so_map.car2car(map_b, map_a)
        prod.data *= map_b_proj.data

    return prod


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
sq_win_alms_dir = "sq_win_alms"

pspy_utils.create_directory(sq_win_alms_dir)

n_sq_alms, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)


log.info(f"number of sq win alms to compute : {n_sq_alms}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_sq_alms - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]

    log.info(f"[{task:02d}] Computing map2alm for {sv1}_{ar1}x{sv2}_{ar2}...")

    win_T1 = so_map.read_map(d["window_T_%s_%s" % (sv1, ar1)])
    win_T2 = so_map.read_map(d["window_T_%s_%s" % (sv2, ar2)])

    sq_win = mult(win_T1, win_T2, log)
    # sq_win = win_T1.copy()
    # sq_win.data[:] *= win_T2.data[:]
    sqwin_alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)

    np.save(f"{sq_win_alms_dir}/alms_{sv1}_{ar1}x{sv2}_{ar2}.npy", sqwin_alm)
