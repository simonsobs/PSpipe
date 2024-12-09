"""
This script compute all alms corresponding to all agora fg components and write them to disk, it
is very similar to get_alms.py except some specific thing to take into account Kristen notation
"""

import sys
import time

import numpy as np
from pixell import enmap
from pspipe_utils import kspace, log, misc, pspipe_list
from pspy import pspy_utils, so_dict, so_map, so_mpi, sph_tools

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
deconvolve_pixwin = d["deconvolve_pixwin"]
niter = d["niter"]
apply_kspace_filter = d["apply_kspace_filter"]

kristen_notation = {}
kristen_notation["dr6_pa4_f220"] = "220_pa4"
kristen_notation["dr6_pa5_f090"] = "90_pa5"
kristen_notation["dr6_pa5_f150"] = "150_pa5"
kristen_notation["dr6_pa6_f090"] = "90_pa6"
kristen_notation["dr6_pa6_f150"] = "150_pa6"
component_path = "/pscratch/sd/k/kmsurrao/ACT_DR6_non_gaussian_sims/individual_components_092924/car_withbeam/"
components = ["anomalous", "cib", "cmb_seed1", "dust", "ksz", "radio", "rksz", "sync", "tsz"]


window_dir = "windows"
alms_dir = "alms_components"
plot_dir = "plot_components"

pspy_utils.create_directory(alms_dir)
pspy_utils.create_directory(plot_dir)

n_ar, sv_list, ar_list = pspipe_list.get_arrays_list(d)

log.info(f"number of arrays for the mpi loop : {n_ar}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_ar-1)
color_range = [250, 30, 30]

for task in subtasks:
    task = int(task)
    sv, ar = sv_list[task], ar_list[task]

    log.info(f"[{task}] Computing alm for '{sv}' survey and '{ar}' array")

    win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
    win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])

    window_tuple = (win_T, win_pol)

    if win_T.pixel == "CAR":
        win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])

        if apply_kspace_filter:
            ks_f = d[f"k_filter_{sv}"]
            filter = kspace.get_kspace_filter(win_T, ks_f)

        inv_pixwin_lxly = None
        if deconvolve_pixwin:
            if d[f"pixwin_{sv}"]["pix"] == "CAR":
                # compute the CAR pixel function in fourier space
                wy, wx = enmap.calc_window(win_T.data.shape, order=d[f"pixwin_{sv}"]["order"])
                inv_pixwin_lxly = (wy[:,None] * wx[None,:]) ** (-1)


    cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]

    t0 = time.time()
    for k, comp in enumerate(components):
        kris_name = kristen_notation[f"{sv}_{ar}"]
        map = f"{component_path}/{comp}_{kris_name}.fits"

        if win_T.pixel == "CAR":
            split = so_map.read_map(map, geometry=win_T.data.geometry)

            if comp in ["radio", "cib"]:
                ps_mask = so_map.read_map(d[f"ps_mask_{sv}_{ar}"])
                split.data *= ps_mask.data

            if apply_kspace_filter:
                log.info(f"[{task}] apply kspace filter on {map}")
                split = kspace.filter_map(split,
                                          filter,
                                          win_kspace,
                                          inv_pixwin=inv_pixwin_lxly,
                                          weighted_filter=ks_f["weighted"],
                                          use_ducc_rfft=True)
                        
            else:
                split.subtract_mono_dipole(window_tuple)
                log.info(f"[{task}] WARNING: no kspace filter is applied on {map} subtract mono and dipole instead")
                if (deconvolve_pixwin) & (inv_pixwin_lxly is not None):
                    split = so_map.fourier_convolution(split,
                                                       inv_pixwin_lxly,
                                                       window=win_kspace,
                                                       use_ducc_rfft=True)
                
                
        elif win_T.pixel == "HEALPIX":
            split = so_map.read_map(map)

        split = split.calibrate(cal=cal, pol_eff=pol_eff)

        master_alms = sph_tools.get_alms(split, window_tuple, niter, lmax)
        np.save(f"{alms_dir}/alms_{sv}_{ar}_{comp}.npy", master_alms)
        
        down_split = split.copy()
        down_split = down_split.downgrade(4)
        down_split.plot(file_name=f"{plot_dir}/{comp}_{sv}_{ar}", color_range=color_range)

        split.data[:] *= win_T.data[:]
        down_split = split.copy()
        down_split = down_split.downgrade(4)
        down_split.plot(file_name=f"{plot_dir}/{comp}_{sv}_{ar}_windowed", color_range=color_range)
        
    log.info(f"[{task}] execution time {time.time() - t0} seconds")
