# This script generates an (e)xtra mask for peculiar structure like big dust blobs. It also add mask
# used by mnms simulation as well as some coordinates stuff to be hidden. The generated mask is then
# used when doing map filtering such as kspace filtering to avoid nasty side effects
import sys

import numpy as np
from pspipe_utils import log, pspipe_list
from pspy import so_dict, so_map

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


def apply_coordinate_mask(mask, coord):
    """
    Apply a mask based on coordinate

    Parameters
    ----------
    mask: so_map
      the mask on which the coordinate mask will be applied
    coord: list of list of float
        format is [[dec0, ra0], [dec1, ra1]]
         we create a rectangle mask from this coordinate
         in the convention assumed in this code
         dec0, ra0 are the coordiante of the top left corner
         dec1, ra1 are the coordinate of the bottom right corner
    """

    dec_ra = np.deg2rad(coord)
    pix1 = mask.data.sky2pix(dec_ra[0])
    pix2 = mask.data.sky2pix(dec_ra[1])
    min_pix = np.min([pix1, pix2], axis=0).astype(int)
    max_pix = np.max([pix1, pix2], axis=0).astype(int)
    mask.data[min_pix[0] : max_pix[0], min_pix[1] : max_pix[1]] = 0

    return mask


n, sv_list, ar_list = pspipe_list.get_arrays_list(d)

for i in range(n):
    sv, ar = sv_list[i], ar_list[i]
    log.info(f"Generate extra mask for '{sv}' survey and '{ar}' array")

    log.info("Apply mnms mask")
    mask = so_map.read_map(d[f"mnmns_mask_{sv}_{ar}"])

    # optionnaly apply an extra coordinate mask
    if coord_mask := d.get(f"coord_mask_{sv}_{ar}"):
        log.info("Apply coord mask")
        mask = apply_coordinate_mask(mask, coord_mask)

    if blobs := d.get(f"blob_mask_{sv}_{ar}"):
        log.info("Apply blob mask")
        for b in blobs:
            mask.data *= so_map.generate_source_mask(mask, np.deg2rad([b[0], b[1]]), b[2] * 60).data

    mask_name = f"act_xtra_mask_{sv}_{ar}"
    log.info(f"Writing {mask_name}...")
    mask.write_map(f"{mask_name}.fits")
    mask.downgrade(4).plot(file_name=mask_name)
