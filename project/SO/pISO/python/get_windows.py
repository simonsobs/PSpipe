"""
This script create the window functions used in the PS computation
They consist of a point source mask, a galactic mask,  a mask based on the amount of cross linking in the data,
and a mask based on pathological pixels (identified during the map based simulation analysis), note that
we also produce a window that include the pixel weighting.
The different masks are apodized.
We also produce a kspace-mask that will later be used for the kspace filtering operation, in order to remove the edges of the survey and avoid nasty pixels.
MODIFIED VERSION, SKIP IVAR AND XLINK
"""

import sys
import os
from os.path import join as opj
import re

import numpy as np
from pixell import enmap
from pspipe_utils import log, pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, so_window


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


surveys = d["surveys"]
# the apodisation length of the final survey mask
apod_kspace_degree = d['apod_kspace_degree']
apod_survey_degree = d["apod_survey_degree"]

# we will skip the edges of the survey where the noise is very difficult to model, the default is to skip 0.5 degree for
# constructing the kspace mask and 2 degree for the final survey mask, this parameter can be used to rescale the default

window_dir = d["window_dir"]
pspy_utils.create_directory(window_dir)

plot_dir = opj(d['plots_dir'], 'windows')
pspy_utils.create_directory(plot_dir)

# Use this if you want to only compute one window (for testing)
# d["surveys"] = ['SO']
# d["arrays_SO"] = ['i1_f090']

# here we list the different windows that need to be computed, we will then do a MPI loops over this list
n_wins, winname_list = pspipe_list.get_windownames_list(d)

log.info(f"number of windows to compute : {n_wins}")

my_masks: dict[so_map.so_map] = {}

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_wins - 1)

def sa(emap):
    return np.sum(emap * emap.pixsizemap())/(4*np.pi)*41253

for task in subtasks:
    winname = winname_list[task]

    log.info(f"[{task}] create windows for '{winname}'")

    # need a template geometry to help load cutouts of masks
    shape, wcs = enmap.read_map_geometry(d[f"template_geometry_{winname}"])
    shape = shape[-2:]
    template_geom = (shape, wcs)

    my_masks["baseline"] = so_map.car_template_from_shape_wcs(1, *template_geom, dtype=np.float32)
    my_masks["baseline"].data = enmap.ones(*template_geom, dtype=np.float32)

    # first get the kspace masks, including possible ra and dec range

    log.info(f"[{task}] apply kspace masks for {winname}")

    kspace_mask = 1
    for kspace_mask_fn in d[f"kspace_masks_{winname}"]:
        kspace_mask *= enmap.read_map(kspace_mask_fn, geometry=template_geom).astype(np.float32, copy=False)
    my_masks["baseline"].data[:] *= kspace_mask

    # compute the distance to the nearest 0
    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * np.pi / 180)
    # here we remove pixels near the edges in order to avoid pixels with very low hit count
    # note the hardcoded 0.5 degree value that can be rescale with the edge_skip_rescale argument in the dictfile.
    my_masks["baseline"].data[dist.data < d['kspace_mask_edge_cut']] = 0

    if "ra_range" in d:
        log.info(f'Applying {d["ra_range"]} RA range')
        ra_range_rad = np.deg2rad(d["ra_range"])
        pos_map = my_masks["baseline"].data.posmap()
        ra_map = pos_map[1]
        ra_mask = (
            (ra_map <= ra_range_rad[0]) | (ra_map >= ra_range_rad[1])
        )
        my_masks["baseline"].data[ra_mask] *= 0

    if "dec_range" in d:
        log.info(f'Applying {d["dec_range"]} DEC range')
        dec_range_rad = np.deg2rad(d["dec_range"])
        pos_map = my_masks["baseline"].data.posmap()
        dec_map = pos_map[0]
        dec_mask = (
            (dec_map <= dec_range_rad[0]) | (dec_map >= dec_range_rad[1])
        )
        my_masks["baseline"].data[dec_mask] *= 0
    
    my_masks["kspace"] = my_masks["baseline"].copy()
    
    # with this we can create the k space mask this will only be used for applying the kspace filter
    log.info(
        f"[{task}] apodize kspace mask with {apod_kspace_degree:.2f} apod and write it to disk"
    )

    # we apodize this k space mask
    my_masks["kspace"] = so_window.create_apodization(
        my_masks["kspace"], "C1", apod_kspace_degree, use_rmax=True
    )
    my_masks["kspace"].data = my_masks["kspace"].data.astype(np.float32, copy=False)
    my_masks["kspace"].write_map(f"{window_dir}/window_{winname}_kspace.fits")

    log.info(f"[{task}] joint kspace mask solid angle: {sa(my_masks["kspace"].data)}")

    # compare to the kspace mask we will skip for the nominal mask
    # an additional 2 degrees to avoid ringing from the filter
    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * np.pi / 180)
    my_masks["baseline"].data[dist.data < d['kspace_to_baseline_edge_cut']] = 0

    log.info(f"[{task}] apodize mask with {apod_survey_degree:.2f} apod")

    # apodisation of the final mask
    my_masks["baseline"] = so_window.create_apodization(
        my_masks["baseline"], "C1", apod_survey_degree, use_rmax=True
    )

    # add on other masks and apodize them, like a point source mask
    if d[f"baseline_masks_{winname}"] is not None:
        for i, baseline_mask_fn in enumerate(d[f"baseline_masks_{winname}"]):
            baseline_mask = so_map.car_template_from_shape_wcs(1, *template_geom, dtype=np.float32)
            baseline_mask.data = enmap.read_map(baseline_mask_fn, geometry=template_geom).astype(np.float32, copy=False)
            baseline_apod = d[f'baseline_apods_{winname}'][i]
            baseline_mask = so_window.create_apodization(
                baseline_mask, "C1", baseline_apod, use_rmax=True
            )
        my_masks["baseline"].data[:] *= baseline_mask.data

    my_masks["baseline"].data = my_masks["baseline"].data.astype(np.float32, copy=False)
    my_masks["baseline"].write_map(f"{window_dir}/window_{winname}_{"baseline"}.fits")

    log.info(f"[{task}] joint baseline mask solid angle: {sa(my_masks["baseline"].data)}")

    # Plot baseline and kspace windows
    for mask_type, mask in my_masks.items():
        log.info(f"[{task}] downgrade and plot {mask_type} ")
        print(f"{plot_dir}/window_{winname}_{mask_type}")
        mask.downgrade(4).plot(file_name=f"{plot_dir}/window_{winname}_{mask_type}")

# for plotting maps
so_mpi.barrier()
n_maps, sv_list, ar_list = pspipe_list.get_arrays_list(d)

subtasks = so_mpi.taskrange(imin=0, imax=n_maps - 1)

if len(d["plot_windowed_maps"]) > 0:
    pspy_utils.create_directory(f"{plot_dir}/windowed_maps")

for task in subtasks:
    sv, ar = sv_list[task], ar_list[task]
    
    if f"{sv}_{ar}" in d["plot_windowed_maps"]:
        window_T = enmap.read_map(d[f'window_T_{sv}_{ar}'])
        if d[f'window_pol_{sv}_{ar}'] != d[f'window_T_{sv}_{ar}']:
            window_pol = enmap.read_map(d[f'window_pol_{sv}_{ar}'])
        else:
            window_pol = window_T

        for s, sv_ar_split in enumerate(d[f"maps_{sv}_{ar}"]):
            maps_to_plot = so_map.read_map(sv_ar_split)
            maps_to_plot.data[0] *= window_T
            maps_to_plot.data[1:] *= window_pol
            maps_to_plot.downgrade(4).calibrate(
                cal=d[f"cal_{sv}_{ar}"], pol_eff=d[f"pol_eff_{sv}_{ar}"]
            ).plot(
                file_name=f"{plot_dir}/windowed_maps/{sv}_{ar}_split{s}",
                color_range=(1000, 300, 300),
            )

# Save the paramfile to keep track
d.write_to_file(f'{window_dir}/_paramfile.dict')
