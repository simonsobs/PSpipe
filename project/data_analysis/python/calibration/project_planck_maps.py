# This script projects the Planck maps onto the ACT survey

import glob
import os
import re
import sys

import numpy as np
from pspy import so_dict, so_map, so_mpi

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

npipe_map_directory = "/global/cfs/cdirs/cmb/data/planck2020/pla/frequency_maps/Multi-detector"
freqs, map_files = [], []
for ar in d.get("arrays_planck", raise_error=True):
    files = glob.glob(os.path.join(npipe_map_directory, "HFI*{}-*full.fits".format(ar[1:])))
    map_files += files
    freqs += len(files) * [ar[1:]]

# Survey mask
survey = so_map.read_map(d.get("survey_planck", raise_error=True))
survey.ncomp = 3
survey.data = np.tile(survey.data, (survey.ncomp, 1, 1))

# Mask dir for removing mon/dipole
masks_dir = os.path.join(d["data_dir"], "planck/likelihood_mask/")
mask_tmpl = os.path.join(masks_dir, "COM_Mask_Likelihood-{}-{}-{}_2048_R3.00.fits")

nmaps = len(map_files)
print(f"number of map to project : {nmaps}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=nmaps - 1)
print(subtasks)

for task in subtasks:
    task = int(task)
    map_file = map_files[task]
    print(f"Reading {map_file}...")
    npipe_map = so_map.read_map(map_file, fields_healpix=(0, 1, 2), coordinate="gal")
    npipe_map.data *= 10 ** 6

    if d.get("remove_mono_dipo_t", True):
        mask_hm1 = so_map.read_map(mask_tmpl.format("temperature", freqs[task], "hm1"))
        mask_hm2 = so_map.read_map(mask_tmpl.format("temperature", freqs[task], "hm2"))
        mask = mask_hm1.copy()
        mask.data *= mask_hm2.data
        npipe_map.data[0] = so_map.subtract_mono_dipole(npipe_map.data[0], mask.data)
    if d.get("remove_mono_dipo_pol", True):
        mask_hm1 = so_map.read_map(mask_tmpl.format("polarization", freqs[task], "hm1"))
        mask_hm2 = so_map.read_map(mask_tmpl.format("polarization", freqs[task], "hm2"))
        mask = mask_hm1.copy()
        mask.data *= mask_hm2.data
        npipe_map.data[1] = so_map.subtract_mono_dipole(npipe_map.data[1], mask.data)
        npipe_map.data[2] = so_map.subtract_mono_dipole(npipe_map.data[2], mask.data)

    print("Projecting in CAR pixellisation...")
    car_project = so_map.healpix2car(npipe_map, survey)
    print("Applying survey mask & convert into float32...")
    car_project.data = car_project.data.astype(np.float32)

    basename = os.path.basename(map_file)
    split_name = "splitA" if "-1" in basename else "splitB"
    car_file = re.sub("(-.*)_2048", "_" + split_name + "_2048", basename)
    car_file = os.path.join(d["data_dir"], "planck_maps", car_file)
    print(f"Storing {car_file}...")
    car_project.write_map(car_file)
