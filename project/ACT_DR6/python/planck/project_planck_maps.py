"""
This script is used to project Planck maps
We project npipe maps found at NERSC:/global/cfs/cdirs/cmb/data/planck2020/npipe
into a CAR pixellization.
We also project legacy maps found at NERSC:/global/cfs/cdirs/cmb/data/planck2018/pr3/frequencymaps
replacing missing pixesl by nppepixels
"""
import sys
import numpy as np
import healpy as hp
from pspy import so_dict, so_map, so_mpi, pspy_utils
from pspipe_utils import log, misc
from pixell import reproject


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


out_dir = "planck_projected"
pspy_utils.create_directory(out_dir)

# Planck frequencies
planck_freqs = ["100", "143", "217", "353"]

release_dir = d["release_dir"]
template_name = f"{release_dir}/maps/published/act_dr6.02_std_AA_night_pa4_f220_4way_set0_map_srcfree.fits"
template = so_map.read_map(template_name)
shape, wcs = template.data.geometry

# Define data directories
npipe_map_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe/"
legacy_map_dir = "/global/cfs/cdirs/cmb/data/planck2018/pr3/frequencymaps/"

splits_npipe = ["A", "B"]
splits_legacy = ["halfmission-1", "halfmission-2"]


# Mono and dipole parameters
# from NPIPE paper https://arxiv.org/pdf/2007.04997.pdf
dip_amp = 3366.6 #uK
l = 263.986
b = 48.247
dipole = dip_amp * hp.pixelfunc.ang2vec((90-b)/180*np.pi, l/180*np.pi)
monopole = {
    "100": -70.,
    "143": -81.,
    "217": -182.4,
    "353": 395.2,
}

n_freqs = len(planck_freqs)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_freqs - 1)


for task in subtasks:
    freq = planck_freqs[int(task)]
    for split_npipe, split_legacy in zip(splits_npipe, splits_legacy):
    
        # Data path npipe
        map_file_npipe = f"{npipe_map_dir}/npipe6v20{split_npipe}/npipe6v20{split_npipe}_{freq}_map.fits"
        log.info(f"[{freq} GHz - split {split_npipe}] Reading map ...")
        npipe_map = so_map.read_map(map_file_npipe, coordinate="gal", fields_healpix=[0,1,2])
        
        npipe_map.data *= 10 ** 6 # Convert from K to uK

        # for npipe you want to subtract the monopole and dipole (in T)
        npipe_map.data[0] = so_map.subtract_mono_dipole(npipe_map.data[0], values=(monopole[freq], dipole))
        
        log.info(f"[{freq} GHz - split {split_npipe}] Projecting in CAR pixellization ...")
        car_project = so_map.healpix2car(npipe_map, template)
        car_project.data = car_project.data.astype(np.float32)

        out_file_name = f"npipe6v20{split_npipe}_f{freq}_map"
        car_project.write_map(file_name=f"{out_dir}/{out_file_name}.fits")
        car_project.downgrade(8).plot(file_name=f"{out_dir}/{out_file_name}",
                                    color_range=[300, 100, 100])
                                    
        
        # Inverse variance npipe
        log.info(f"[{freq} GHz - split {split_npipe}] Reading ivar map ...")
        var_file = misc.str_replace(map_file_npipe, "map.fits", "wcov_mcscaled.fits")
        # Read the variance map
        npipe_ivar = so_map.read_map(var_file, coordinate="gal", fields_healpix=[0])
        # Convert into inverse variance in uK
        npipe_ivar.data[npipe_ivar.data != 0] = 1 / (1e12 * npipe_ivar.data[npipe_ivar.data != 0])

        rot=f"{npipe_ivar.coordinate},{template.coordinate}"
        ivar_project = reproject.healpix2map(npipe_ivar.data, shape[1:], wcs, rot=rot, method="spline",
                                             spin=[0], extensive=True)
        ivar_project = so_map.from_enmap(ivar_project)

        out_file_name = f"npipe6v20{split_npipe}_f{freq}_ivar"
        ivar_project.write_map(file_name=f"{out_dir}/{out_file_name}.fits")
        ivar_project.downgrade(8).plot(file_name=f"{out_dir}/{out_file_name}",
                                       color_range=5e-4)




        # now do legacy
        legacy_file = f"{legacy_map_dir}/HFI_SkyMap_{freq}_2048_R3.01_{split_legacy}.fits"
        legacy_map = so_map.read_map(legacy_file, coordinate="gal", fields_healpix=[0,1,2])
        id = np.where(legacy_map.data[:] == hp.pixelfunc.UNSEEN)
        legacy_map.data *= 10 ** 6 # Convert from K to uK
        
        # replace missing pixels in legacy by npipe pixell
        legacy_map.data[id] = npipe_map.data[id]

        log.info(f"[{freq} GHz - split {split_legacy}] Projecting in CAR pixellization ...")
        car_project = so_map.healpix2car(legacy_map, template)
        car_project.data = car_project.data.astype(np.float32)

        out_file_name = f"HFI_SkyMap_2048_R3.01_{split_legacy}_f{freq}_map"
        car_project.write_map(file_name=f"{out_dir}/{out_file_name}.fits")

        car_project.downgrade(8).plot(file_name=f"{out_dir}/{out_file_name}",
                                    color_range=[300, 100, 100])


        # Inverse variance legacy
        log.info(f"[{freq} GHz - split {split_legacy}] Reading ivar map ...")
        legacy_ivar = so_map.read_map(legacy_file, coordinate="gal", fields_healpix=[4])
        print(legacy_ivar.data.shape)
        # Convert into inverse variance in uK
        legacy_ivar.data[legacy_ivar.data != 0] = 1 / (1e12 * legacy_ivar.data[legacy_ivar.data != 0])

        rot=f"{legacy_ivar.coordinate},{template.coordinate}"
        ivar_project = reproject.healpix2map(legacy_ivar.data, shape[1:], wcs, rot=rot, method="spline",
                                             spin=[0], extensive=True)
        ivar_project = so_map.from_enmap(ivar_project)

        out_file_name = f"HFI_SkyMap_2048_R3.01_{split_legacy}_f{freq}_ivar"
        ivar_project.write_map(file_name=f"{out_dir}/{out_file_name}.fits")
        ivar_project.downgrade(8).plot(file_name=f"{out_dir}/{out_file_name}",
                                       color_range=5e-4)
