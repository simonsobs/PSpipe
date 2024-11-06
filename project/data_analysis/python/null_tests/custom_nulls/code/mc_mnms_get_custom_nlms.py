description = """
generates noise alms for custom nulls
"""
from pspy import pspy_utils, so_dict, so_mpi
from mnms import noise_models as nm
from pixell import curvedsky
from pspipe_utils import log
import numpy as np
import time
import sys
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--paramfile", type=str,
                    help="Filename (full or relative path) of paramfile to use")
parser.add_argument("--iStart", type=int, default=None)
parser.add_argument("--iStop", type=int, default=None)
parser.add_argument("--survey", type=str, default=None)
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)


lmax = d["lmax"]
survey = args.survey
iStart = args.iStart
iStop = args.iStop

all_surveys = ["el1", "el2", "el3", "pwv1", "pwv2", "t1", "t2", "inout"]
sim_num_start = {}
for count, sv in enumerate(all_surveys):
    sim_num_start[sv] = count * 1000

log.info(f"generating sims from {iStart} to {iStop} for survey: {survey}, seed start at {sim_num_start[survey]}")

# Set the lmax corresponding to a pre-computed noise model
lmax_noise_sim = 10800 # Should be moved to the paramfile ?

# Aliases for arrays
arrays_alias = {
    "pa4": {"pa4a": "pa4_f150", "pa4b": "pa4_f220"},
    "pa5": {"pa5a": "pa5_f090", "pa5b": "pa5_f150"},
    "pa6": {"pa6a": "pa6_f090", "pa6b": "pa6_f150"}
}

wafers = ["pa4", "pa5", "pa6"]
# Load the noise models
noise_models = {}
for wafer_name in wafers:
    if survey in ["el1", "el2", "el3"]:
        noise_models[wafer_name] = nm.BaseNoiseModel.from_config(f"act_dr6v4_el_split",
                                                                 d[f"noise_sim_type_{wafer_name}"],
                                                                 *arrays_alias[wafer_name].keys(),
                                                                 el_split=[survey])
    if survey in ["pwv1", "pwv2"]:
        noise_models[wafer_name] = nm.BaseNoiseModel.from_config(f"act_dr6v4_pwv_split",
                                                                 d[f"noise_sim_type_{wafer_name}"],
                                                                 *arrays_alias[wafer_name].keys(),
                                                                 pwv_split=[survey])
    if survey in ["t1", "t2"]:
        arrays_alias["pa4"] = {"pa4b": "pa4_f220"}
        noise_models[wafer_name] = nm.BaseNoiseModel.from_config(f"act_dr6v4_t_split",
                                                                 d[f"noise_sim_type_{wafer_name}"],
                                                                 *arrays_alias[wafer_name].keys(),
                                                                 t_split=[survey])


    if survey == "inout":
        noise_models[wafer_name] = nm.BaseNoiseModel.from_config("act_dr6v4_inout_split",
                                                                 d[f"noise_sim_type_{wafer_name}"],
                                                                 *arrays_alias[wafer_name].keys(),
                                                                 inout_split=["inout1", "inout2"])

if survey == "inout":
    arrays_alias["pa4"] = {"qid1": "pa4_f150_in",  "qid2": "pa4_f220_in",
                           "qid3": "pa4_f150_out", "qid4": "pa4_f220_out"}
    arrays_alias["pa5"] = {"qid1": "pa5_f090_in",  "qid2": "pa5_f150_in",
                           "qid3": "pa5_f090_out", "qid4": "pa5_f150_out"}
    arrays_alias["pa6"] = {"qid1": "pa6_f090_in",  "qid2": "pa6_f150_in",
                           "qid3": "pa6_f090_out", "qid4": "pa6_f150_out"}


# Create output dir
nlms_dir = f"mnms_noise_alms_{survey}"
pspy_utils.create_directory(nlms_dir)

n_splits = 2 if survey in ["t1", "t2"] else 4
mpi_list = []
for id_split in range(n_splits):
    mpi_list.append(id_split)

# we will use mpi over the number of splits
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(mpi_list)-1)
for id_mpi in subtasks:
    k = mpi_list[id_mpi]
    t1 = time.time()
    for wafer_name in wafers:
        for iii in range(iStart, iStop + 1):
            t2 = time.time()

            sim_arrays = noise_models[wafer_name].get_sim(split_num=k,
                                                          sim_num=sim_num_start[survey] + iii,
                                                          lmax=lmax_noise_sim,
                                                          alm=False,
                                                          keep_model=True)

            log.info(f"[Sim n° {iii}] {survey} {wafer_name} split {k} noise realization generated in {time.time()-t2:.2f} s")

            for i, (qid, ar) in enumerate(arrays_alias[wafer_name].items()):

                cal, pol_eff = d[f"cal_{survey}_{ar}"], d[f"pol_eff_{survey}_{ar}"]

                sim_arrays[i, 0, :][0] *= cal
                sim_arrays[i, 0, :][1] *= cal / pol_eff
                sim_arrays[i, 0, :][2] *= cal / pol_eff

                noise_alms = curvedsky.map2alm(sim_arrays[i, 0, :], lmax=lmax)

                np.save(f"{nlms_dir}/mnms_nlms_{survey}_{ar}_set{k}_{iii:05d}.npy", noise_alms)

        noise_models[wafer_name].cache_clear()

    log.info(f"split {k} - {iStop-iStart+1} sims generated in {time.time()-t1:.2f} s")
