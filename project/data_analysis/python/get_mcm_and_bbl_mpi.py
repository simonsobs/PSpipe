"""
This script computes the mode coupling matrices and the binning matrices Bbl
for the different surveys and arrays
"""

from pspy import so_map, so_mcm, pspy_utils, so_dict, so_mpi
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

mcm_dir = "mcms"
pspy_utils.create_directory(mcm_dir)

surveys = d["surveys"]
lmax = d["lmax"]

if d["use_toeplitz"] == True:
    print("we will use the toeplitz approximation")
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None


sv1_list, ar1_list, sv2_list, ar2_list = [], [], [], []
n_mcms = 0
for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):
                # This ensures that we do not repeat redundant computations
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                sv1_list += [sv1]
                ar1_list += [ar1]
                sv2_list += [sv2]
                ar2_list += [ar2]
                n_mcms += 1
                
print("number of mcm matrices to compute : %s" % n_mcms)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_mcms - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]
    
    print("%s_%s x %s_%s" % (sv1, ar1, sv2, ar2))
    
    l, bl1 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv1, ar1)])
    win1_T = so_map.read_map(d["window_T_%s_%s" % (sv1, ar1)])
    win1_pol = so_map.read_map(d["window_pol_%s_%s" % (sv1, ar1)])
    
    l, bl2 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv2, ar2)])
    win2_T = so_map.read_map(d["window_T_%s_%s" % (sv2, ar2)])
    win2_pol = so_map.read_map(d["window_pol_%s_%s" % (sv2, ar2)])

    mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=(win1_T, win1_pol),
                                                win2=(win2_T, win2_pol),
                                                bl1=(bl1, bl1),
                                                bl2=(bl2, bl2),
                                                binning_file=d["binning_file"],
                                                niter=d["niter"],
                                                lmax=d["lmax"],
                                                type=d["type"],
                                                l_exact=l_exact,
                                                l_band=l_band,
                                                l_toep=l_toep,
                                                save_file="%s/%s_%sx%s_%s"%(mcm_dir, sv1, ar1, sv2, ar2))



