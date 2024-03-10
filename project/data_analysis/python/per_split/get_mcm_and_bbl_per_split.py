"""
This script computes the mode coupling matrices and the binning matrices Bbl
for the different surveys and arrays.
"""

import sys

from pspipe_utils import log, pspipe_list, misc
from pspy import pspy_utils, so_dict, so_map, so_mcm, so_mpi

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

mcm_dir = "mcms"
pspy_utils.create_directory(mcm_dir)

surveys = d["surveys"]
lmax = d["lmax"]
binned_mcm = d["binned_mcm"]

if d["use_toeplitz_mcm"] == True:
    log.info("we will use the toeplitz approximation")
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

n_mcms, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)

log.info(f"number of mcm matrices to compute : {n_mcms}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_mcms - 1)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]

    n_splits_1 = d[f"n_splits_{sv1}"]
    n_splits_2 = d[f"n_splits_{sv2}"]

    beam_per_split_T1 = d[f"beam_T_{sv1}_{ar1}_per_split"]
    window_per_split_T1 = d[f"window_T_{sv1}_{ar1}_per_split"]
    
    beam_per_split_pol1 = d[f"beam_pol_{sv1}_{ar1}_per_split"]
    window_per_split_pol1 = d[f"window_pol_{sv1}_{ar1}_per_split"]

    beam_per_split_T2 = d[f"beam_T_{sv2}_{ar2}_per_split"]
    window_per_split_T2 = d[f"window_T_{sv2}_{ar2}_per_split"]

    beam_per_split_pol2 = d[f"beam_pol_{sv2}_{ar2}_per_split"]
    window_per_split_pol2 = d[f"window_pol_{sv2}_{ar2}_per_split"]


    for s1 in range(n_splits_1):
    
        l, bl1 = misc.read_beams(beam_per_split_T1[s1], beam_per_split_pol1[s1])
        win1_T = so_map.read_map(window_per_split_T1[s1])
        win1_pol = so_map.read_map(window_per_split_pol1[s1])

        for s2 in range(n_splits_2):
            if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue

            log.info(f"[{task:02d}] mcm matrix for {sv1}_{ar1}_{s1} x {sv2}_{ar2}_{s2}")
            
            l, bl2 = misc.read_beams(beam_per_split_T2[s2], beam_per_split_pol2[s2])
            win2_T = so_map.read_map(window_per_split_T2[s2])
            win2_pol = so_map.read_map(window_per_split_pol2[s2])

            mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=(win1_T, win1_pol),
                                                        win2=(win2_T, win2_pol),
                                                        bl1=(bl1["T"], bl1["E"]),
                                                        bl2=(bl2["T"], bl2["E"]),
                                                        binning_file=d["binning_file"],
                                                        niter=d["niter"],
                                                        lmax=d["lmax"],
                                                        type=d["type"],
                                                        l_exact=l_exact,
                                                        l_band=l_band,
                                                        l_toep=l_toep,
                                                        binned_mcm=binned_mcm,
                                                        save_file=f"{mcm_dir}/{sv1}_{ar1}_{s1}x{sv2}_{ar2}_{s2}")
