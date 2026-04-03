"""
This script compute the one dimensional power spectra of spt simulations
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi
from pspipe_utils import pspipe_list, log
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
binning_file = d["binning_file"]
binned_mcm = d["binned_mcm"]
release_dir = d["release_dir"]
alm_conv = d[f"alm_conv_{survey}"]


mcm_dir = "mcms"
tf_dir = "sim_spectra_for_tf"

pspy_utils.create_directory(tf_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
arrays_spt = d["arrays_spt"]


so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")

    t0 = time.time()
    
    master_alms = {}
    
    for ar in arrays_spt:
    
        sim_in = so_map.read_map(f"{release_dir}/simulated_maps/input_maps/input_maps_realization{iii:03d}_{ar}ghz.fits")
        sim_out = so_map.read_map(f"{release_dir}/simulated_maps/output_maps/masking_yes/output_maps_masking_yes_realization{iii:03d}_{ar}ghz.fits")

        win_T = so_map.read_map(d[f"window_T_{survey}_{ar}"])
        win_pol = so_map.read_map(d[f"window_pol_{survey}_{ar}"])

        window_tuple = (win_T, win_pol)

        master_alms[survey, ar, "nofilter"] = sph_tools.get_alms(sim_in, window_tuple, niter, lmax, alm_conv=alm_conv)
        master_alms[survey, ar, "filter"] = sph_tools.get_alms(sim_out, window_tuple, niter, lmax, alm_conv=alm_conv)
        
        alm_mask = hp.read_alm(d[f"alm_mask_{survey}_{ar}"], hdu=1)
        alm_mask = hp.sphtfunc.resize_alm(alm_mask, d["lmax_mask"], d["lmax_mask"], lmax, lmax)

        master_alms[survey, ar, "nofilter_mask"] = master_alms[survey, ar, "nofilter"] * alm_mask
        master_alms[survey, ar, "filter_mask"] = master_alms[survey, ar, "filter"] * alm_mask


    _, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
        
    n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
        
    for i_spec in range(n_spec):
        sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]
        spec_name = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}"

        mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}", spin_pairs=spin_pairs)

        for filt in ["filter", "filter_mask", "nofilter", "nofilter_mask"]:

            l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, filt], master_alms[sv2, ar2, filt], spectra=spectra)

            lb, ps = so_spectra.bin_spectra(l,
                                            ps_master,
                                            binning_file,
                                            lmax,
                                            type=type,
                                            mbb_inv=mbb_inv,
                                            spectra=spectra,
                                            binned_mcm=binned_mcm)

            so_spectra.write_ps(tf_dir + f"/{spec_name}_{filt}_{iii:05d}.dat", lb, ps, type, spectra=spectra)

    log.info(f"[{iii}]  Simulation n° {iii:05d} done in {time.time()-t0:.02f} s")
