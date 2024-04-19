"""
This script is used to get the noise alms for Planck NPIPE or legacy
noise simulations and to save them to disk in a format
compatible with the requirements of `mc_mnms_get_spectra.py`
Disk usage: 144M per alm
"""
from pspy import so_dict, pspy_utils, so_mpi, sph_tools, so_map
from pixell import curvedsky, reproject
from pspipe_utils import log
import numpy as np
import time
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

lmax = d["lmax"]
niter = d["niter"]
sv = "Planck"
planck_arrays = d[f"arrays_{sv}"]

version = d["planck_version"]

npipe_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe"
legacy_dir = "/pscratch/sd/t/tlouis/data_analysis_v4_dec/planck_co_analysis/planck_legacy_sim/"

if version == "npipe":
    planck_splits = ["A", "B"]
if version == "legacy":
    planck_splits = ["hm1", "hm2"]

output_dir = "noise_alms"
pspy_utils.create_directory(output_dir)

n_sims = d["iStop"] - d["iStart"] + 1

mpi_list = [(f, s, iii) for f in planck_arrays for s in planck_splits for iii in range(d["iStart"], d["iStop"]+1)]

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(mpi_list)-1)

log.info(f"Number of maps to project : {len(mpi_list)}")
for id_mpi in subtasks:

    t0 = time.time()

    ar, split, iii = mpi_list[id_mpi]
    freq = d[f"freq_info_{sv}_{ar}"]["freq_tag"]

    cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]

    if version == "npipe":
        # Note that and we are using the sim index 0 as the first simulation (corresponding to the 200th NPIPE sim.)
        map_name = f"{npipe_dir}/npipe6v20{split}_sim/{iii+200:04d}/residual/residual_npipe6v20{split}_{freq}_{iii+200:04d}.fits"
        map_name_noise_fix  = f"{npipe_dir}/npipe6v20{split}_sim/{iii+200:04d}/noisefix/noisefix_{freq}{split}_{iii+200:04d}.fits"

        hp_map = so_map.read_map(map_name, coordinate="gal", fields_healpix=[0,1,2])
        noise_fix = so_map.read_map(map_name_noise_fix, coordinate="gal", fields_healpix=[0,1,2])

        hp_map.data[:] += noise_fix.data[:]

    if version == "legacy":
        map_name = f"{legacy_dir}/legacy_noise_sim_{freq}_{split}_{iii:05d}.fits"
        hp_map = so_map.read_map(map_name, coordinate="gal", fields_healpix=[0,1,2])

        
    hp_map.data *= 1e6 # from K to uK

    alms = sph_tools.map2alm(hp_map, niter=niter, lmax=lmax, dtype=np.complex64)

    alms[0] *= cal
    alms[1] *= cal / pol_eff
    alms[2] *= cal / pol_eff

    # Rotate alms
    rot=f"{hp_map.coordinate},equ"
    curvedsky.rotate_alm(alms, *reproject.rot2euler(rot), inplace=True)

    # Note that we use the PSpipe conventions : i.e. indexing splits with an integer (here 0,1)
    np.save(f"{output_dir}/nlms_{sv}_f{freq}_set{planck_splits.index(split)}_{iii:05d}.npy", alms)

    log.info(f"[Planck {version} {freq}{split} sim n°{iii:05d}] Saved to disk in {time.time()-t0:.2f} s")

