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
# the apodisation length of the point source mask in degree
apod_pts_source_degree = d["apod_pts_source_degree"]
# the apodisation length of the final survey mask
apod_survey_degree = d["apod_survey_degree"]
apod_kspace_degree = d['apod_kspace_degree']

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

for task in subtasks:
    winname = winname_list[task]

    log.info(f"[{task}] create windows for '{winname}'")

    # need a template geometry to help load cutouts of masks. we get this by 
    # looking for a map that will use this window
    windowkey_pattern = 'window_(T|pol|kspace)_(.*)'
    windowname_pattern = f'window_{winname}_(kspace|baseline)'
    for k, v in d.items():
        match_key = re.search(windowkey_pattern, k)
        if match_key is not None: # this is a window key
            windowname_text = os.path.splitext(os.path.basename(v))[0] # e.g. 'window_lat_iso_i1_kspace'
            match_val = re.search(windowname_pattern, windowname_text) # this window key uses this window
            if match_val is not None:
                template_mapname = match_key.group(2) # this is the mapname in the window key
                break
    
    shape, wcs = enmap.read_map_geometry(d[f"maps_{template_mapname}"][0]) # first split
    shape = shape[-2:]
    template_geom = (shape, wcs)

    my_masks["baseline"] = so_map.car_template_from_shape_wcs(1, shape, wcs)
    my_masks["baseline"].data = enmap.ones(*template_geom, dtype=np.float32)

    if d.get(f"extra_masks_{winname}") is not None:
        log.info(f"[{task}] apply extra masks")

        def sa(emap):
            return np.sum(emap * emap.pixsizemap())/(4*np.pi)*41253
        
        extra_mask = 1
        for extra_mask_fn in d[f"extra_masks_{winname}"]:
            _extra_mask = enmap.read_map(extra_mask_fn, geometry=template_geom)
            log.info(f"[{task}] extra mask {extra_mask_fn} solid angle: {sa(_extra_mask)}")
            extra_mask *= _extra_mask
        log.info(f"[{task}] joint extra mask solid angle: {sa(extra_mask)}")
        my_masks["baseline"].data[:] *= extra_mask

    # compute the distance to the nearest 0
    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * np.pi / 180)
    # here we remove pixels near the edges in order to avoid pixels with very low hit count
    # note the hardcoded 0.5 degree value that can be rescale with the edge_skip_rescale argument in the dictfile.
    my_masks["baseline"].data[dist.data < d['extra_mask_edge_cut']] = 0

    # apply the galactic mask
    log.info(f"[{task}] apply galactic mask")

    gal_mask = so_map.read_map(d[f"gal_mask_{winname}"], geometry=template_geom)
    my_masks["baseline"].data *= gal_mask.data

    # with this we can create the k space mask this will only be used for applying the kspace filter
    log.info(
        f"[{task}] appodize kspace mask with {apod_kspace_degree:.2f} apod and write it to disk"
    )
    my_masks["kspace"] = my_masks["baseline"].copy()

    # we apodize this k space mask with a 1 degree apodisation

    my_masks["kspace"] = so_window.create_apodization(
        my_masks["kspace"], "C1", apod_kspace_degree, use_rmax=True
    )
    my_masks["kspace"].data = my_masks["kspace"].data.astype(np.float32)
    my_masks["kspace"].write_map(f"{window_dir}/window_{winname}_kspace.fits")

    # compare to the kspace mask we will skip for the nominal mask
    # an additional 2 degrees to avoid ringing from the filter

    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * np.pi / 180)
    my_masks["baseline"].data[dist.data < d['kspace_mask_edge_cut']] = 0

    for mask_type in ["baseline"]:
        # optionnaly apply a patch mask

        log.info(f"[{task}] apodize mask with {apod_survey_degree:.2f} apod")

        # apodisation of the final mask
        my_masks[mask_type] = so_window.create_apodization(
            my_masks[mask_type], "C1", apod_survey_degree, use_rmax=True
        )

        # create a point source mask and apodise it

        log.info(f"[{task}] include ps mask")

        ps_mask = so_map.read_map(d[f"ps_mask_{winname}"], geometry=template_geom)
        ps_mask = so_window.create_apodization(
            ps_mask, "C1", apod_pts_source_degree, use_rmax=True
        )
        my_masks[mask_type].data *= ps_mask.data

        my_masks[mask_type].data = my_masks[mask_type].data.astype(np.float32, copy=False)

        if mask_type == "baseline":
            Omega = so_window.get_survey_solid_angle(my_masks[mask_type])
            Omega_srad = Omega / (4 * np.pi) * 41253
            log.info(f"[{task}] {winname} baseline mask solid angle: {Omega_srad}")

        my_masks[mask_type].write_map(f"{window_dir}/window_{winname}_{mask_type}.fits")

    for mask_type, mask in my_masks.items():
        log.info(f"[{task}] downgrade and plot {mask_type} ")
        print(f"{plot_dir}/window_{winname}_{mask_type}")
        mask.downgrade(4).plot(file_name=f"{plot_dir}/window_{winname}_{mask_type}")

# for plotting maps
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
                color_range=(300, 100, 100),
            )

# Save the paramfile to keep track
d.write_to_file(f'{window_dir}/_paramfile.dict')
