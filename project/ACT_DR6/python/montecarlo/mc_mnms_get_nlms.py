"""
T.B.D.
"""
from pspy import pspy_utils, so_dict, so_mpi
from mnms import noise_models as nm
from pixell import curvedsky
from pspipe_utils import log
import numpy as np
import time
import argparse

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--iStart", help="Set starting index of simultions", type=int)
parser.add_argument("--iStop", help="Set stopping index of simulations", type=int)
parser.add_argument("--bunch", help="Set bunch index", type=int)
parser.add_argument("--nbunch", help="Set number of simulation per bunch", default=50, type=int)
args, dict_file = parser.parse_known_args()

d = so_dict.so_dict()
d.read_from_file(dict_file[0])
log = log.get_logger(**d)

surveys = ["dr6"]
lmax = d["lmax"]



# Set the lmax corresponding to a pre-computed noise model
lmax_noise_sim = 10800 # Should be moved to the paramfile ?

# Aliases for arrays
arrays_alias = {
    "pa4": {"pa4a": "pa4_f150", "pa4b": "pa4_f220"},
    "pa5": {"pa5a": "pa5_f090", "pa5b": "pa5_f150"},
    "pa6": {"pa6a": "pa6_f090", "pa6b": "pa6_f150"}
}

# Load the noise models
noise_models = {
    wafer_name: nm.BaseNoiseModel.from_config("act_dr6v4", d[f"noise_sim_type_{wafer_name}"], *arrays_alias[wafer_name].keys())
    for sv in surveys for wafer_name in sorted({ar.split("_")[0] for ar in d[f"arrays_{sv}"]})
}

# Create output dir
nlms_dir = "noise_alms"
pspy_utils.create_directory(nlms_dir)

arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
n_splits = {sv: d[f"n_splits_{sv}"] for sv in surveys}

mpi_list = []
for sv in surveys:
    log.info(f"Running with {n_splits[sv]} splits for survey {sv}")
    for id_split in range(n_splits[sv]):
        mpi_list.append((sv, id_split))

iStart = args.iStart or d["iStart"]
iStop = args.iStop or d["iStop"]
if args.bunch is not None:
    iStart = int(args.bunch * args.nbunch)
    iStop = int((args.bunch + 1) * args.nbunch) - 1

# we will use mpi over the number of splits
so_mpi.init(True)
#subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])
subtasks = so_mpi.taskrange(imin=0, imax=len(mpi_list)-1)
for id_mpi in subtasks:

    sv, k = mpi_list[id_mpi]

    t0 = time.time()

    wafers = sorted({ar.split("_")[0] for ar in arrays[sv]})

    t1 = time.time()
    for wafer_name in wafers:

        for iii in range(iStart, iStop+1):

            t2 = time.time()
            sim_arrays = noise_models[wafer_name].get_sim(split_num=k,
                                                                sim_num=iii,
                                                                lmax=lmax_noise_sim,
                                                                alm=False,
                                                                keep_model=True)

            log.info(f"[Sim n° {iii}] {wafer_name} split {k} noise realization generated in {time.time()-t2:.2f} s")
            for i, (qid, ar) in enumerate(arrays_alias[wafer_name].items()):

                cal, pol_eff = d[f"cal_{sv}_{ar}"], d[f"pol_eff_{sv}_{ar}"]
                sim_arrays[i, 0, :][0] *= cal
                sim_arrays[i, 0, :][1] *= cal / pol_eff
                sim_arrays[i, 0, :][2] *= cal / pol_eff
                noise_alms = curvedsky.map2alm(sim_arrays[i, 0, :], lmax=lmax)

                np.save(f"{nlms_dir}/nlms_{sv}_{ar}_set{k}_{iii:05d}.npy", noise_alms)

        noise_models[wafer_name].cache_clear()

    log.info(f"split {k} {iStop-iStart+1} sims generated in {time.time()-t1:.2f} s")
