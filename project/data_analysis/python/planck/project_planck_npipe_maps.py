"""
This script is used to project Planck npipe maps
found at NERSC:/global/cfs/cdirs/cmb/data/planck2020/npipe
into a CAR pixellization.
Main usage: These maps are needed to perform calibration of
ACT DR6 maps, to estimate the large-scale power loss and to
assess consistency against Planck.
"""
import sys
import numpy as np
from pspy import so_dict, so_map, so_mpi, pspy_utils
from pspipe_utils import log
from pixell import reproject

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

out_dir = "npipe_projected"
pspy_utils.create_directory(out_dir)

# Planck frequencies
planck_freqs = d.get("arrays_Planck", raise_error=True)
planck_freqs = [f.replace("f", "") for f in planck_freqs]

# Survey mask
survey = so_map.read_map(d.get("survey_Planck", raise_error=True))
shape, wcs = survey.data.geometry
survey = so_map.car_template_from_shape_wcs(3, shape, wcs)

# Define data directories
npipe_map_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe"

# Define a map list
splits = {"A": "hm1", "B": "hm2"}
map_names = []
for split in splits:
    for freq in planck_freqs:
        map_names.append((freq, split))

# Mono and dipole parameters
dip_amp = 3366.6 #uK
l = 263.986
b = 48.247
dipole = dip_amp * hp.pixelfunc.ang2vec((90-b)/180*np.pi, l/180*np.pi)
monopole = {
    "100":-70.,
    "143":-81.,
    "217":-182.4,
    "353":395.2,
}

n_maps = len(map_names)
log.info(f"number of map to project : {n_maps}")

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_maps - 1)

for task in subtasks:
    task = int(task)

    freq, split = map_names[task]

    # Data path
    map_file = f"{npipe_map_dir}/npipe6v20{split}/npipe6v20{split}_{freq}_map.fits"

    # Load map
    log.info(f"[{freq} GHz - split {split}] Reading map ...")
    npipe_map = so_map.read_map(map_file, coordinate="gal", fields_healpix=[0,1,2])
    npipe_map.data *= 10 ** 6 # Convert from K to uK

    # Subtract mono and dipole (temperature)
    npipe_map.data[0] = so_map.subtract_mono_dipole(npipe_map.data[0], mono=monopole[freq], dipole=dipole)


    log.info(f"[{freq} GHz - split {split}] Projecting in CAR pixellization ...")
    car_project = so_map.healpix2car(npipe_map, survey)
    car_project.data = car_project.data.astype(np.float32)

    out_file_name = f"npipe6v20{split}_f{freq}_map"
    car_project.write_map(file_name=f"{out_dir}/{out_file_name}.fits")

    car_project.downgrade(8).plot(file_name=f"{out_dir}/{out_file_name}",
                                  color_range=[300, 100, 100])

    # Inverse variance
    log.info(f"[{freq} GHz - split {split}] Reading ivar map ...")
    npipe_ivar = so_map.read_map(map_file.replace("map.fits", "wcov_mcscaled.fits"), coordinate="gal", fields_healpix=[0])
    npipe_ivar.data[npipe_ivar.data != 0] = 1 / (1e12 * npipe_ivar.data)

    ivar_project = reproject.enmap_from_healpix_interp(npipe_ivar.data, shape[1:], wcs)
    ivar_project = so_map.from_enmap(ivar_project)

    out_file_name = f"npipe6v20{split}_f{freq}_ivar"
    ivar_project.write_map(file_name=f"{out_dir}/{out_file_name}.fits")
