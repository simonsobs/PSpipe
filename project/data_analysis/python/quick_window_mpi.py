# This script create the window functions used in the PS computation
# They consist of a point source mask, a galactic mask and a mask based on the amount of cross linking in the data
# The different masks are apodized.
# We also produce a binary mask that will later be used for the kspace filtering operation, in order to remove the edges and avoid nasty pixels before
# this not so well defined Fourier operation.

import sys

import numpy as np
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

# the apodisation lenght of the point source mask in degree
apod_pts_source_degree = d["apod_pts_source_degree"]
# the apodisation lenght of the survey x gal x cross linking mask
apod_survey_degree = d["apod_survey_degree"]
# we will skip the edges of the survey where the noise is very difficult to model
skip_from_edges_degree = d["skip_from_edges_degree"]
# the threshold on the amount of cross linking to keep the data in
cross_link_threshold = d["cross_link_threshold"]

window_dir = "windows"
surveys = d["surveys"]

pspy_utils.create_directory(window_dir)
ps_mask = so_map.read_map(d["ps_mask"])
gal_mask = so_map.read_map(d["gal_mask"])

patch = None
if "patch" in d:
    patch = so_map.read_map(d["patch"])


# here we list the different windows that need to be computed, we will then do a MPI loops over this list
sv_list, ar_list = [], []
n_wins = 0
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        sv_list += [sv]
        ar_list += [ar]
        n_wins += 1

print("number of windows to compute : %s" % n_wins)
so_mpi.init(True)

subtasks = so_mpi.taskrange(imin=0, imax=n_wins - 1)
print(subtasks)
for task in subtasks:

    task = int(task)
    sv, ar = sv_list[task], ar_list[task]

    survey_mask = gal_mask.copy()
    survey_mask.data[:] = 1

    maps = d["maps_%s_%s" % (sv, ar)]

    for k, map in enumerate(maps):
        print(map)
        map = so_map.read_map(map)
        survey_mask.data[map.data[0] == 0.0] = 0.0

    for k, map in enumerate(maps):
        index = map.find("map_srcfree.fits")
        xlink_map = map[:index] + "xlink.fits"
        print(xlink_map)
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
    binary.write_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
    binary = binary.downgrade(4)
    binary.plot(file_name="%s/binary_%s_%s" % (window_dir, sv, ar))

    # Now we create the final window function that will be used in the analysis
    survey_mask.data[dist.data < skip_from_edges_degree] = 0
    survey_mask = so_window.create_apodization(survey_mask, "C1", apod_survey_degree, use_rmax=True)
    ps_mask = so_window.create_apodization(ps_mask, "C1", apod_pts_source_degree, use_rmax=True)
    survey_mask.data *= ps_mask.data

    survey_mask.write_map("%s/window_%s_%s.fits" % (window_dir, sv, ar))
    survey_mask = survey_mask.downgrade(4)
    survey_mask.plot(file_name="%s/window_%s_%s" % (window_dir, sv, ar))
