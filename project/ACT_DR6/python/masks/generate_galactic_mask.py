# This script projects the Planck galactic masks onto the ACT survey

import os
import re
import sys

import healpy as hp
import numpy as np
from astropy.io import fits
from pspy import so_dict, so_map
from pspipe_utils import log

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


mask_dir = "galactic_masks"

# Survey mask
survey = so_map.read_map(d["template"])
if survey.ncomp > 2:
    # Only use temperature
    survey.data = survey.data[0]
survey.ncomp = 1

# Planck galatic masks
fn = d["planck_galactic_masks"]
hdul = fits.open(fn)

# Try to get nside and apodization from filename
m = re.search("apo(.)", fn)
apod = int(m.group(1)) if m else d.get("galactic_mask_apodization", 0)
m = re.search("apo.*_(.*)_", fn)
nside = int(m.group(1)) if m else d.get("galactic_mask_nside", 2048)

# Loop over maps
data = hdul[1].data
for mask_name in data.names:
    log.info(f"Processing {mask_name} mask...")
    healpix_map = so_map.healpix_template(
        1, nside, coordinate=d.get("galatic_mask_coordinate", "gal")
    )
    healpix_map.data = (
        hp.reorder(np.float64(data.field(mask_name)), n2r=True)
        if d.get("galactic_mask_nest_ordering", True)
        else data.field(mask_name)
    )

    log.info("Projecting in CAR pixellisation...")
    car_project = so_map.healpix2car(healpix_map, survey)
    car_project.data[car_project.data > 0.5] = 1
    car_project.data[car_project.data <= 0.5] = 0

    car_file = os.path.join(
        mask_dir, f"mask_galactic_equatorial_car_{mask_name.lower()}_apo{apod}.fits"
    )
    log.info(f"Writing {car_file}...")
    car_project.write_map(car_file)
    car_project.downgrade(4).plot(file_name=car_file.replace(".fits", ""))

hdul.close()
