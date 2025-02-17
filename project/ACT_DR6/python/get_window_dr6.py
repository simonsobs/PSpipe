"""
This script create the window functions used in the PS computation
They consist of a point source mask, a galactic mask,  a mask based on the amount of cross linking in the data,
and a mask based on pathological pixels (identified during the map based simulation analysis), note that
we also produce a window that include the pixel weighting.
The different masks are apodized.
We also produce a kspace-mask that will later be used for the kspace filtering operation, in order to remove the edges of the survey and avoid nasty pixels.
"""

import sys

import numpy as np
from pixell import enmap
from pspipe_utils import log, pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, so_window


def create_crosslink_mask(xlink_map, cross_link_threshold):
    """
    Create a mask to remove pixels with a small amount of x-linking
    We compute this using product from the map maker which assess the amount
    of scan direction that hits each pixels in the map
    the product have 3 component and we compute sqrt(Q ** 2 + U ** 2) / I by analogy with the polarisation fraction
    A high value of this quantity means low level of xlinking, we mask all pixels above a given threshold
    note that the mask is designed on a downgraded version of the maps, this is to avoid small scale structure in the mask
    Parameters
    ----------
    xlink_map: so_map
      3 component so_map assessing the direction of scan hitting each pixels
    cross_link_threshold: float
      a threshold above which region of the sky are considered not x-linked
    """

    xlink = so_map.read_map(xlink_map)
    xlink_lowres = xlink.downgrade(32)
    with np.errstate(invalid="ignore"):
        x_mask = (
            np.sqrt(xlink_lowres.data[1] ** 2 + xlink_lowres.data[2] ** 2) / xlink_lowres.data[0]
        )
    x_mask[np.isnan(x_mask)] = 1
    x_mask[x_mask >= cross_link_threshold] = 1
    x_mask[x_mask < cross_link_threshold] = 0
    x_mask = 1 - x_mask

    xlink_lowres.data[0] = x_mask
    xlink = so_map.car2car(xlink_lowres, xlink)
    x_mask = xlink.data[0].copy()
    id = np.where(x_mask > 0.9)
    x_mask[:] = 0
    x_mask[id] = 1
    return x_mask


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


surveys = d["surveys"]
# the apodisation length of the point source mask in degree
apod_pts_source_degree = d["apod_pts_source_degree"]
# the apodisation length of the final survey mask
apod_survey_degree = d["apod_survey_degree"]
# the threshold on the amount of cross linking to keep the data in
cross_link_threshold = d["cross_link_threshold"]
# pixel weight with inverse variance above n_ivar * median are set to ivar
# this ensure that the window is not totally dominated by few pixels with too much weight
n_med_ivar = d["n_med_ivar"]

# we will skip the edges of the survey where the noise is very difficult to model, the default is to skip 0.5 degree for
# constructing the kspace mask and 2 degree for the final survey mask, this parameter can be used to rescale the default
rescale = d["edge_skip_rescale"]

window_dir = "windows"

pspy_utils.create_directory(window_dir)

patch = None
if "patch" in d:
    patch = so_map.read_map(d["patch"])


# here we list the different windows that need to be computed, we will then do a MPI loops over this list
# we also force surveys to be dr6 (otherwise product will be missing)
d["surveys"] = ["dr6"]
n_wins, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of windows to compute : {n_wins}")

my_masks = {}

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_wins - 1)

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]

    log.info(f"[{task}] create windows for '{sv}' survey and '{ar}' array...")

    gal_mask = so_map.read_map(d[f"gal_mask_{sv}_{ar}"])

    my_masks["baseline"] = gal_mask.copy()
    my_masks["baseline"].data[:] = 1

    maps = d[f"maps_{sv}_{ar}"]

    ivar_all = gal_mask.copy()
    ivar_all.data[:] = 0

    # the first step is to iterate on the maps of a given array to identify pixels with zero values
    # we will form a first survey mask as the union of all these split masks
    # we also compute the average ivar map

    log.info(f"[{task}] create survey mask from ivar maps")

    for k, map in enumerate(maps):
        if d[f"src_free_maps_{sv}"] == True:
            index = map.find("map_srcfree.fits")
        else:
            index = map.find("map.fits")

        ivar_map = map[:index] + "ivar.fits"
        ivar_map = so_map.read_map(ivar_map)
        my_masks["baseline"].data[ivar_map.data[:] == 0.0] = 0.0
        ivar_all.data[:] += ivar_map.data[:]

    ivar_all.data[:] /= np.max(ivar_all.data[:])

    if d[f"extra_mask_{sv}_{ar}"] is not None:
        log.info(f"[{task}] apply extra mask")
        extra_mask = so_map.read_map(d[f"extra_mask_{sv}_{ar}"])
        my_masks["baseline"].data[:] *= extra_mask.data[:]

    log.info(
        f"[{task}] compute distance to the edges and remove {0.5*rescale:.2f} degree from the edges"
    )
    # compute the distance to the nearest 0
    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * rescale * np.pi / 180)
    # here we remove pixels near the edges in order to avoid pixels with very low hit count
    # note the hardcoded 0.5 degree value that can be rescale with the edge_skip_rescale argument in the dictfile.
    my_masks["baseline"].data[dist.data < 0.5 * rescale] = 0

    # apply the galactic mask
    log.info(f"[{task}] apply galactic mask")
    my_masks["baseline"].data *= gal_mask.data

    # with this we can create the k space mask this will only be used for applying the kspace filter
    log.info(f"[{task}] appodize kspace mask with {rescale:.2f} apod and write it to disk")
    my_masks["kspace"] = my_masks["baseline"].copy()

    # we apodize this k space mask with a 1 degree apodisation
    
    my_masks["kspace"] = so_window.create_apodization(my_masks["kspace"], "C1", 1 * rescale, use_rmax=True)
    my_masks["kspace"].data = my_masks["kspace"].data.astype(np.float32)
    my_masks["kspace"].write_map(f"{window_dir}/window_{sv}_{ar}_kspace.fits")

    # compare to the kspace mask we will skip for the nominal mask
    # an additional 2 degrees to avoid ringing from the filter

    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline"].data[dist.data < 2 * rescale] = 0

    # now we create a xlink mask based on the xlink threshold

    log.info(f"[{task}] create xlink mask")

    my_masks["xlink"] = my_masks["baseline"].copy()
    for k, map in enumerate(maps):
        if d[f"src_free_maps_{sv}"] == True:
            index = map.find("map_srcfree.fits")
        else:
            index = map.find("map.fits")

        xlink_map = map[:index] + "xlink.fits"
        x_mask = create_crosslink_mask(xlink_map, cross_link_threshold)
        my_masks["xlink"].data *= x_mask

    for mask_type in ["baseline", "xlink"]:
        # optionnaly apply a patch mask

        if patch is not None:
            log.info(f"[{task}] apply patch mask")

            my_masks[mask_type].data *= patch.data

        log.info(f"[{task}] apodize mask with {apod_survey_degree:.2f} apod")

        # apodisation of the final mask
        my_masks[mask_type] = so_window.create_apodization(my_masks[mask_type], "C1", apod_survey_degree, use_rmax=True)
   
        # create a point source mask and apodise it

        log.info(f"[{task}] include ps mask")

        ps_mask = so_map.read_map(d[f"ps_mask_{sv}_{ar}"])
        ps_mask = so_window.create_apodization(ps_mask, "C1", apod_pts_source_degree, use_rmax=True)
        my_masks[mask_type].data *= ps_mask.data

        my_masks[mask_type].data = my_masks[mask_type].data.astype(np.float32)
        
        if mask_type == "baseline":
            Omega = so_window.get_survey_solid_angle(my_masks[mask_type])
            Omega_srad = Omega / (4 * np.pi) * 41253
            log.info(f"[{task}] {sv} {ar} baseline mask solid angle: {Omega_srad}")

        my_masks[mask_type].write_map(f"{window_dir}/window_{sv}_{ar}_{mask_type}.fits")

        # we also make a version of the windows taking into account the ivar of the maps
        log.info(f"[{task}] include ivar ")

        mask_type_w = mask_type + "_ivar"

        my_masks[mask_type_w] = my_masks[mask_type].copy()
        id = np.where(ivar_all.data[:] * my_masks[mask_type_w].data[:] != 0)
        med = np.median(ivar_all.data[id])
        ivar_all.data[ivar_all.data[:] > n_med_ivar * med] = n_med_ivar * med
        my_masks[mask_type_w].data[:] *= ivar_all.data[:]

        my_masks[mask_type_w].data = my_masks[mask_type_w].data.astype(np.float32)
        my_masks[mask_type_w].write_map(f"{window_dir}/window_{sv}_{ar}_{mask_type_w}.fits")

    for mask_type, mask in my_masks.items():
        log.info(f"[{task}] downgrade and plot {mask_type} ")
        mask = mask.downgrade(4)
        mask.plot(file_name=f"{window_dir}/window_{sv}_{ar}_{mask_type}")

