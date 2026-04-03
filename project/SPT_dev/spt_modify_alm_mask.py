"""
script to modify spt3g alm mask
"""
import sys

import numpy as np
import healpy as hp
from pspipe_utils import log
from pspy import pspy_utils, so_dict, sph_tools



def create_mask_lm(l, m, l_cutoff, m_cutoff, width=20):
    m_mask = np.where(m <= (m_cutoff - width), 1.0,
             np.where(m >= m_cutoff, 0.0,
             0.5 * (1 + np.cos(np.pi * (m - (m_cutoff - width)) / width))))
    l_mask = np.where(l <= l_cutoff, 0.0,
             np.where(l >= (l_cutoff + width), 1.0,
             0.5 * (1 - np.cos(np.pi * (l - l_cutoff) / width))))

    return l_mask * m_mask



d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


lmax_mask = d["lmax_mask"]
surveys = d["surveys"]
release_dir = d["release_dir"]
plot_dir = "plots"
pspy_utils.create_directory(plot_dir)


for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
        alm_mask = hp.read_alm(release_dir + f"ancillary_products/specific_to_c25/alm_mask_{ar}ghz.fits", hdu=(1,2,3))
        sph_tools.show_alm_triangle(alm_mask, lmax_mask, 0, 1, xlims=[0,2000], ylims=[0,2000], fig_file=f"{plot_dir}/old_alm_mask")

        l, m = hp.Alm.getlm(lmax_mask)
        
        soft_mask = create_mask_lm(l, m, l_cutoff=400, m_cutoff=270, width=50)
        alm_mask *= (1.0 - soft_mask)
        
        sph_tools.show_alm_triangle(alm_mask, lmax_mask, 0, 1, xlims=[0,2000], ylims=[0,2000], fig_file=f"{plot_dir}/new_alm_mask")

        hp.fitsfunc.write_alm(f"alm_mask_{ar}ghz_modif.fits", alm_mask, overwrite=True)
