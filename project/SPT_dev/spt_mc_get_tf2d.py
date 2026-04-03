"""
This script uses spt simulations to compute 2d (l,m) transfer functions
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mpi
from pspipe_utils import log
import numpy as np
import healpy as hp
import sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

survey = "spt"
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
release_dir = d["release_dir"]
alm_conv = d[f"alm_conv_{survey}"]


tf_dir = "tf2d"

pspy_utils.create_directory(tf_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
arrays_spt = d["arrays_spt"]


so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")

    master_alms = {}
    
    for ar in arrays_spt:
    
        sim_in = so_map.read_map(f"{release_dir}/simulated_maps/input_maps/input_maps_realization{iii:03d}_{ar}ghz.fits")
        sim_out = so_map.read_map(f"{release_dir}/simulated_maps/output_maps/masking_yes/output_maps_masking_yes_realization{iii:03d}_{ar}ghz.fits")

        win_T = so_map.read_map(d[f"window_T_{survey}_{ar}"])
        
        fsky = np.sum(win_T.data) / (12 * win_T.nside ** 2)

        win_pol = so_map.read_map(d[f"window_pol_{survey}_{ar}"])

        window_tuple = (win_T, win_pol)

        master_alms["nofilter"] = sph_tools.get_alms(sim_in, window_tuple, niter, lmax, alm_conv=alm_conv)
        master_alms["filter"] = sph_tools.get_alms(sim_out, window_tuple, niter, lmax, alm_conv=alm_conv)
        
        l = np.arange(lmax)
        for i, comp in enumerate(["TT", "EE", "BB"]):
            for filt in ["nofilter", "filter"]:
            
                master_alms[filt][i] = hp.sphtfunc.almxfl(master_alms[filt][i], l)
                ps_2d = np.abs(master_alms[filt][i] * np.conjugate(master_alms[filt][i]))/(2 * np.pi * fsky)
                
                np.save(f"{tf_dir}/tf2d_{comp}_{ar}_{filt}_{iii:05d}.npy", ps_2d)
                
