"""
This script create the window functions used in the PS computation
They consist of a point source mask, a galactic mask,  a mask based on the amount of cross linking in the data, and a coordinate mask, note that
we also produce a window that include the pixel weighting.
The different masks are apodized.
We also produce a kspace-mask that will later be used for the kspace filtering operation, in order to remove the edges of the survey and avoid nasty pixels.
"""

import sys

import numpy as np
from pixell import enmap
from pspipe_utils import log, pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, so_window
from pspipe_utils import pspipe_list
from pixell import enmap
import scipy
from scipy import ndimage


def create_crosslink_mask(xlink_map, cross_link_threshold):
    """
    Create a mask to remove pixels with a small amount of x-linking
    We compute this using product from the map maker which assess the amount
    of scan direction that hits each pixels in the map
    the product have 3 component and we compute sqrt(Q **2 + U ** 2) / I by analogy with the polarisation fraction
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
        x_mask = (np.sqrt(xlink_lowres.data[1] ** 2 + xlink_lowres.data[2] ** 2) / xlink_lowres.data[0])
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

window_dir = "test_windows"

pspy_utils.create_directory(window_dir)

patch = None
if "patch" in d:
    patch = so_map.read_map(d["patch"])

# here we list the different windows that need to be computed, we will then do a MPI loops over this list
n_wins, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of windows to compute : {n_wins}")

my_masks = {}

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_wins - 1)

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]
   
    log.info(f"[{task}] create windows for '{sv}' survey and '{ar}' array...")

    # using same gal mask for everything
    #gal_mask = so_map.read_map(d[f"gal_mask_{sv}_{ar}"])
    gal_mask = so_map.read_map(d[f"gal_mask"])

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

# NO EXTRA MASK FOR NOW

#    if d[f"extra_mask_{sv}_{ar}"] is not None:
#        log.info(f"[{task}] apply extra mask")
#        extra_mask = so_map.read_map(d[f"extra_mask_{sv}_{ar}"])
#        my_masks["baseline"].data[:] *= extra_mask.data[:]

    # N/S
    if sv[-5:] == 'north':
        my_masks["baseline"].data[(my_masks["baseline"].data.posmap()[0]<(my_masks["baseline"].data.posmap()[1]*(-23./18.)+(124.*np.pi/180.))) & (my_masks["baseline"].data.posmap()[0]<(my_masks["baseline"].data.posmap()[1]*(23./18.)+(103.5*np.pi/180.)))] = 0
    elif sv[-5:] == 'south':
        my_masks["baseline"].data[(my_masks["baseline"].data.posmap()[0]>(my_masks["baseline"].data.posmap()[1]*(-23./18.)+(124.*np.pi/180.))) | (my_masks["baseline"].data.posmap()[0]>(my_masks["baseline"].data.posmap()[1]*(23./18.)+(103.5*np.pi/180.)))] = 0

    log.info(
        f"Find and fill small holes in the survey mask"
    )
    
#    my_masks[mask_type].data = my_masks[mask_type].data.astype(np.float32)
    my_masks["baseline"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar.fits")
    
    # find & fill holes
    mask_with_holes = my_masks["baseline"].copy()
    mask_without_holes = my_masks["baseline"].copy()
    
    # get intial filled mask, where we fill the holes then skip some distance from the edges
    mask_fill = my_masks["baseline"].copy()
    mask_fill.data[:] = ndimage.binary_fill_holes(mask_fill.data[:])
    
    my_masks["baseline_fill"] = mask_fill.copy()
    my_masks["baseline_fill"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill.fits")
    
    dist_fill = so_window.get_distance(mask_fill, rmax=4 * rescale * np.pi / 180)
    mask_fill.data[dist_fill.data < 0.5 * rescale] = 0
    
    # we want to keep the outer edge from the non-filled mask
    # this accomplishes that (even though it maybe looks a little weird)
    all_ones = mask_fill.copy()
    all_ones.data[:] = 1
    mask_without_holes.data *= (all_ones.data - mask_fill.data)
    mask_without_holes.data += mask_fill.data
    
    my_masks["baseline_fill_reedge"] = mask_without_holes.copy()
    my_masks["baseline_fill_reedge"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill_reedge.fits")
    
    
    # try enlarging holes first
    mask_holefill = my_masks["baseline"].copy()
    
    dist_holefill = so_window.get_distance(mask_holefill, rmax=4 * rescale * np.pi / 180)
    mask_holefill.data[dist_holefill.data < 0.5 * rescale] = 0
    mask_holefill.data[:] = ndimage.binary_fill_holes(mask_holefill.data[:])
    
    my_masks["baseline_holefill"] = mask_holefill.copy()
    my_masks["baseline_holefill"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_smalledgeskip_fill.fits")


#    if d[f"fill_holes_in_windows"] == True:
#        my_masks["baseline"].data[:] = mask_without_holes.data[:]
#        # these are the holes we have effectively filled (if True above) in the final window:
#        coords_holes = np.where((mask_without_holes.data[:] == 1.0) & (mask_with_holes.data[:]==0.0))
#        num_pix_hole = len(coords_holes[0])
#        coords_holes_list = [[coords_holes[0][i],coords_holes[1][i]] for i in range(num_pix_hole)]
#        
#        print(f'window {sv} {ar}')
#        print('number of holes: %s' % num_pix_hole)
#        
#        if d[f"output_hole_coords"] == True:
#            output_hole_dir = d[f"output_hole_dir"]
#            pspy_utils.create_directory(output_hole_dir)
#            np.save(f"{output_hole_dir}/holes_window_{sv}_{ar}.npy", coords_holes_list)

    log.info(
        f"[{task}] compute distance to the edges and remove {0.5*rescale:.2f} degree from the edges"
    )
    # compute the distance to the nearest 0
    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline"].data[dist.data < 0.5 * rescale] = 0
    my_masks["baseline"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_smalledgeskip.fits")
    
    dist = so_window.get_distance(my_masks["baseline_fill"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline_fill"].data[dist.data < 0.5 * rescale] = 0
    my_masks["baseline_fill"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill_smalledgeskip.fits")
    
    dist = so_window.get_distance(my_masks["baseline_fill_reedge"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline_fill_reedge"].data[dist.data < 0.5 * rescale] = 0
    my_masks["baseline_fill_reedge"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill_reedge_smalledgeskip.fits")
    
    
    # apply the galactic mask
    log.info(f"[{task}] apply galactic mask")
    my_masks["baseline"].data *= gal_mask.data
    my_masks["baseline_fill"].data *= gal_mask.data
    my_masks["baseline_fill_reedge"].data *= gal_mask.data
    my_masks["baseline_holefill"].data *= gal_mask.data
    
    my_masks["baseline"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_smalledgeskip_gal.fits")
    my_masks["baseline_fill"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill_smalledgeskip_gal.fits")
    my_masks["baseline_fill_reedge"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill_reedge_smalledgeskip_gal.fits")
    my_masks["baseline_holefill"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_smalledgeskip_fill_gal.fits")
    
    # with this we can create the k space mask this will only be used for applying the kspace filter
#    log.info(f"[{task}] appodize kspace mask with {rescale:.2f} apod and write it to disk")
#    my_masks["kspace"] = my_masks["baseline"].copy()

    # we apodize this k space mask with a 1 degree apodisation
    
#    my_masks["kspace"] = so_window.create_apodization(my_masks["kspace"], "C1", 1 * rescale, use_rmax=True)
#    my_masks["kspace"].data = my_masks["kspace"].data.astype(np.float32)
#    my_masks["kspace"].write_map(f"{window_dir}/kspace_mask_{sv}_{ar}.fits")

    # compare to the kspace mask we will skip for the nominal mask
    # an additional 2 degrees to avoid ringing from the filter

    dist = so_window.get_distance(my_masks["baseline"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline"].data[dist.data < 2 * rescale] = 0
    my_masks["baseline"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_smalledgeskip_gal_bigedgeskip.fits")

    dist = so_window.get_distance(my_masks["baseline_fill"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline_fill"].data[dist.data < 2 * rescale] = 0
    my_masks["baseline_fill"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill_smalledgeskip_gal_bigedgeskip.fits")

    dist = so_window.get_distance(my_masks["baseline_fill_reedge"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline_fill_reedge"].data[dist.data < 2 * rescale] = 0
    my_masks["baseline_fill_reedge"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_fill_reedge_smalledgeskip_gal_bigedgeskip.fits")
    
    dist = so_window.get_distance(my_masks["baseline_holefill"], rmax=4 * rescale * np.pi / 180)
    my_masks["baseline_holefill"].data[dist.data < 2 * rescale] = 0
    my_masks["baseline_holefill"].write_map(f"{window_dir}/window_{sv}_{ar}_baseline_gal_ivar_smalledgeskip_fill_gal_bigedgeskip.fits")

