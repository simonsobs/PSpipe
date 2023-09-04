"""
This script is used to get the noise alms for Planck NPIPE
noise simulations and to save them to disk in a format
compatible with the requirements of `mc_mnms_get_spectra.py`
Disk usage: 144M per alm
"""
from pspy import so_dict, pspy_utils, so_mpi
from pspipe_utils import log
import healpy as hp
import numpy as np
import time
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

lmax = d["lmax"]
sv = "Planck"
planck_arrays = d[f"arrays_{sv}"]
planck_splits = ["A", "B"]

npipe_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe"

output_dir = "noise_alms"
pspy_utils.create_directory(output_dir)

n_sims = d["iStop"] - d["iStart"] + 1

mpi_list = [(f, s, iii) for f in planck_arrays for s in planck_splits for iii in range(n_sims)]

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(mpi_list)-1)

log.info(f"Number of maps to project : {len(mpi_list)}")
for id_mpi in subtasks:

    t0 = time.time()

    ar, split, iii = mpi_list[id_mpi]
    freq = d[f"freq_info_{sv}_{ar}"]["freq_tag"]

    # Note that and we are using the sim index 0 as the first simulation (corresponding to the 200th NPIPE sim.)
    map_name = f"{npipe_dir}/npipe6v20{split}_sim/{iii:04d}/residual/residual_npipe6v20{split}_{freq}_{iii+200:04d}.fits"

    hp_map = hp.read_map(map_name, field=(0,1,2))
    hp_map *= 1e6 # from K to uK

    alms = hp.map2alm(hp_map, lmax=lmax)

    # Note that we use the PSpipe conventions : i.e. indexing splits with an integer (here 0,1)
    np.save(f"{output_dir}/nlms_{sv}_f{freq}_set{planck_splits.index(split)}_{iii:05d}.npy", alms)

    log.info(f"[NPIPE {freq}{split} sim n°{iii:05d}] Saved to disk in {time.time()-t0:.2f} s")
