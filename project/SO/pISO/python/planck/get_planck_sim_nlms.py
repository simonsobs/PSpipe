"""
This script is used to get the noise alms for Planck NPIPE or legacy
noise simulations and to save them to disk in a format
compatible with the requirements of `mc_mnms_get_spectra.py`
"""
from pspy import so_mpi
from pspipe_utils import log
from mnms import utils

import numpy as np
from pixell import curvedsky, reproject
import healpy as hp

import argparse
import time
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--planck-sim-dir", required=True, type=str,
                    help="Directory containing the Planck noise simulations")
parser.add_argument("--out-sim-dir", required=True, type=str,
                    help="Directory in which to save the output alms")
args = parser.parse_args()

log = log.get_logger()

planck_dir = args.planck_sim_dir
out_dir = args.out_sim_dir
os.makedirs(out_dir, exist_ok=True)

planck_svs = ['legacy', 'npipe']
planck_maps = ['100', '143', '217', '353']
lmax = 5400 # all HFI planck nside=2048
niter = 3
sv2info = dict(
    npipe=dict(
        planck_splits=["A", "B"],
        sim_iterator=list(range(300, 800)),
        map_fname_template="{planck_dir}/npipe6v20{planck_split}_sim/{iii:04d}/residual/residual_npipe6v20{planck_split}_{m}_{iii:04d}.fits",
        noise_fix_fname_template="{planck_dir}/npipe6v20{planck_split}_sim/{iii:04d}/noisefix/noisefix_{m}{planck_split}_{iii:04d}.fits",
        noise_model_name='npipe_residual'
    ),
    legacy=dict(
        planck_splits=["hm1", "hm2"],
        sim_iterator=list(range(0, 300)),
        map_fname_template="{planck_dir}/legacy_sims/legacy_noise_sim_{m}_{planck_split}_{iii:05d}.fits",
        noise_fix_fname_template=None,
        noise_model_name='legacy_noise'
    )
)

### 

tasklist = [
    (sv, m, k, iii) 
    for sv in planck_svs
    for m in planck_maps
    for k in sv2info[sv]['planck_splits']
    for iii in sv2info[sv]['sim_iterator']
    ]

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(tasklist)-1)

log.info(f"[Rank {so_mpi.rank}] Number of maps to project : {len(subtasks)}")
for t in subtasks:

    (sv, m, k, iii) = tasklist[t]

    t0 = time.time()

    map_fname_template = sv2info[sv]['map_fname_template']
    map_fname = map_fname_template.format(
        planck_dir=planck_dir,
        planck_split=k,
        m=m,
        iii=iii
    )
    hp_map = hp.read_map(map_fname, field=(0, 1, 2))

    noise_fix_fname_template = sv2info[sv]['noise_fix_fname_template']
    if noise_fix_fname_template is not None:
        noise_fix_fname = noise_fix_fname_template.format(
            planck_dir=planck_dir,
            planck_split=k,
            m=m,
            iii=iii
        )
        hp_map += hp.read_map(noise_fix_fname, field=(0, 1, 2))

    assert not (hp_map == hp.UNSEEN).any(), \
        f'{sv=}, {m=}, {k=}, {iii=}'

    hp_map *= 1e6 # from K to uK

    alm = hp.sphtfunc.map2alm(hp_map, lmax=lmax, iter=niter)

    # Rotate alms
    rot="gal,equ"
    curvedsky.rotate_alm(alm, *reproject.rot2euler(rot), inplace=True)
    
    alm = alm.astype(np.complex64, copy=False)

    # Note that we use the PSpipe conventions : i.e. indexing splits with an integer (here 0,1)
    config_name = f'planck_e2e_raw'
    noise_model_name = sv2info[sv]['noise_model_name']
    qid_names = m

    planck_splits = sv2info[sv]['planck_splits']
    num_splits = len(planck_splits)
    split_num = planck_splits.index(k)
    alm_str = 'alm'

    sim_iterator = sv2info[sv]['sim_iterator']
    sim_num = sim_iterator.index(iii)
    out_fname = f'{out_dir}/{config_name}_{noise_model_name}_{qid_names}_lmax{lmax}_{num_splits}way_set{split_num}_noise_sim_{alm_str}{sim_num:04}.fits'

    utils.write_alm(out_fname, alm)

    log.info(f"[Planck {sv} {m}{k} sim n°{iii:05d}] Saved to disk in {time.time()-t0:.2f} s")