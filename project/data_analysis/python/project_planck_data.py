# This script projects the Planck maps onto the ACT survey

import glob
import os
import re
import sys

import healpy as hp
import numpy as np
from pspy import so_dict, so_map, so_mpi


def subtract_mono_di(map_in, nside=2048):
    mono, dipole = hp.pixelfunc.fit_dipole(map_in)
    print(mono, dipole)
    m = map_in.copy()
    npix = hp.nside2npix(nside)
    bunchsize = npix // 24
    for ibunch in range(npix // bunchsize):
        ipix = np.arange(ibunch * bunchsize, (ibunch + 1) * bunchsize)
        ipix = ipix[(np.isfinite(m.flat[ipix]))]
        x, y, z = hp.pix2vec(nside, ipix, False)
        m.flat[ipix] -= dipole[0] * x
        m.flat[ipix] -= dipole[1] * y
        m.flat[ipix] -= dipole[2] * z
        m.flat[ipix] -= mono
    return m


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

npipe_map_directory = "/global/cfs/cdirs/cmb/data/planck2020/pla/frequency_maps/Multi-detector"
map_files = []
for ar in d.get("arrays_planck", raise_error=True):
    map_files += glob.glob(os.path.join(npipe_map_directory, "HFI*{}-*full.fits".format(ar[1:])))

survey = so_map.read_map(d.get("survey_planck", raise_error=True))
survey.data = np.tile(survey.data, (3, 1, 1))
survey.ncomp = 3

nmaps = len(map_files)
print("number of map to project : {}".format(nmaps))
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=nmaps - 1)
print(subtasks)

for task in subtasks:
    task = int(task)
    map_file = map_files[task]
    print("Reading {}...".format(map_file))
    npipe_map = so_map.read_map(map_file, fields_healpix=(0, 1, 2), coordinate="gal")
    npipe_map.data *= 10 ** 6

    if d.get("remove_mono_dipo_t", False):
        mask = npipe_map.copy()
        mask.data = mask.data[0]
        mask.data[mask.data != 0] = 1
        npipe_map.data[0] = subtract_mono_di(npipe_map.data[0], nside=npipe_map.nside)

    print("Projecting in CAR pixellisation...")
    car_project = so_map.healpix2car(npipe_map, survey)
    print("Applying survey mask & convert into float32...")
    # car_project.data *= survey.data
    car_project.data = car_project.data.astype(np.float32)

    basename = os.path.basename(map_file)
    split_name = "splitA" if "-1" in basename else "splitB"
    car_file = re.sub("(-.*)_2048", "_" + split_name + "_2048", basename)
    car_file = os.path.join(d["data_dir"], "maps", car_file)
    print("Storing {}...".format(car_file))
    car_project.write_map(car_file)
