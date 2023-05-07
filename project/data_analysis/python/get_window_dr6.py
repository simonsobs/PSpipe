"""
This script create the window functions used in the PS computation
They consist of a point source mask, a galactic mask and a mask based on the amount of cross linking in the data, we also produce a window that include the pixel weighting.
The different masks are apodized.
We also produce a binary mask that will later be used for the kspace filtering operation, in order to remove the edges and avoid nasty pixels before
this not so well defined Fourier operation.
"""

import sys

import numpy as np
from pspipe.log import get_logger
from pspipe_utils import pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, so_window


def create_crosslink_mask(xlink_map, cross_link_threshold):
    # remove pixels with very little amount of cross linking
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
log = get_logger(**d)

# the apodisation lenght of the point source mask in degree
apod_pts_source_degree = d["apod_pts_source_degree"]
# the apodisation lenght of the survey x gal x cross linking mask
apod_survey_degree = d["apod_survey_degree"]
# we will skip the edges of the survey where the noise is very difficult to model
skip_from_edges_degree = d["skip_from_edges_degree"]
# the threshold on the amount of cross linking to keep the data in
cross_link_threshold = d["cross_link_threshold"]
# pixel weight with inverse variance above n_ivar * median are set to ivar
# this ensure that the window is not totally dominated by few pixels with too much weight
n_med_ivar = d["n_med_ivar"]

window_dir = "windows"
surveys = d["surveys"]

pspy_utils.create_directory(window_dir)

patch = None
if "patch" in d:
    patch = so_map.read_map(d["patch"])

# here we list the different windows that need to be computed, we will then do a MPI loops over this list
n_wins, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of windows to compute : {n_wins}")
so_mpi.init(True)

subtasks = so_mpi.taskrange(imin=0, imax=n_wins - 1)
for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
    log.info(f"[{task}] create windows for '{sv}' survey and '{ar}' array...")

    gal_mask = so_map.read_map(d[f"gal_mask_{sv}_{ar}"])

    survey_mask = gal_mask.copy()
    survey_mask.data[:] = 1

    maps = d[f"maps_{sv}_{ar}"]

    ivar_all = gal_mask.copy()
    ivar_all.data[:] = 0

    for k, map in enumerate(maps):
        if d[f"src_free_maps_{sv}"] == True:
            index = map.find("map_srcfree.fits")
        else:
            index = map.find("map.fits")

        ivar_map = map[:index] + "ivar.fits"
        log.info(f"using '{ivar_map}' file")
        ivar_map = so_map.read_map(ivar_map)
        survey_mask.data[ivar_map.data[:] == 0.0] = 0.0
        ivar_all.data[:] += ivar_map.data[:]

    ivar_all.data[:] /= np.max(ivar_all.data[:])

    for k, map in enumerate(maps):
        if d[f"src_free_maps_{sv}"] == True:
            index = map.find("map_srcfree.fits")
        else:
            index = map.find("map.fits")

        xlink_map = map[:index] + "xlink.fits"
        log.info(f"using '{xlink_map}' file")
        x_mask = create_crosslink_mask(xlink_map, cross_link_threshold)
        survey_mask.data *= x_mask

    survey_mask.data *= gal_mask.data

    if patch is not None:
        survey_mask.data *= patch.data

    dist = so_window.get_distance(survey_mask, rmax=apod_survey_degree * np.pi / 180)

    # so here we create a binary mask this will only be used in order to skip the edges before applying the kspace filter
    # this step is a bit arbitrary and preliminary, more work to be done here

    binary = survey_mask.copy()
    # Note that we don't skip the edges as much for the binary mask
    # compared to what we will do with the final window, this should prevent some aliasing from the kspace filter to enter the data
    binary.data[dist.data < skip_from_edges_degree / 2] = 0

    binary.data = binary.data.astype(np.float32)
    binary.write_map(f"{window_dir}/binary_{sv}_{ar}.fits")

    # Now we create the final window function that will be used in the analysis
    survey_mask.data[dist.data < skip_from_edges_degree] = 0
    survey_mask = so_window.create_apodization(survey_mask, "C1", apod_survey_degree, use_rmax=True)
    ps_mask = so_map.read_map(d[f"ps_mask_{sv}_{ar}"])
    ps_mask = so_window.create_apodization(ps_mask, "C1", apod_pts_source_degree, use_rmax=True)
    survey_mask.data *= ps_mask.data

    survey_mask.data = survey_mask.data.astype(np.float32)
    survey_mask.write_map(f"{window_dir}/window_{sv}_{ar}.fits")

    # We also create an optional window which also include pixel weighting
    # Note that with use the threshold n_ivar * med so that pixels with very high
    # hit count do not dominate

    survey_mask_weighted = survey_mask.copy()
    id = np.where(ivar_all.data[:] * survey_mask.data[:] != 0)
    med = np.median(ivar_all.data[id])
    ivar_all.data[ivar_all.data[:] > n_med_ivar * med] = n_med_ivar * med
    survey_mask_weighted.data[:] *= ivar_all.data[:]

    survey_mask_weighted.data = survey_mask_weighted.data.astype(np.float32)
    survey_mask_weighted.write_map(f"{window_dir}/window_w_{sv}_{ar}.fits")

    # plot
    binary = binary.downgrade(4)
    binary.plot(file_name=f"{window_dir}/binary_{sv}_{ar}")

    survey_mask = survey_mask.downgrade(4)
    survey_mask.plot(file_name=f"{window_dir}/window_{sv}_{ar}")

    survey_mask_weighted = survey_mask_weighted.downgrade(4)
    survey_mask_weighted.plot(file_name=f"{window_dir}/window_w_{sv}_{ar}")
