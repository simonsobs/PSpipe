description = """
This script stitches together the previously produced lowest-level split-
averaged covariance blocks (at the pseudospectrum level) into array-level
powerspectrum blocks. To do so it sandwiches two pseudo-to-spec operators
that capture the PSpipe, including mode-decoupling, binning (with possible)
Dl factors, and kspace deconvolving.

Users may submit this script using any multinode, multitask
call to sbatch, but they must do so in one call to sbatch, not multiple as in 
a bash loop. If it makes sense to submit multiple distinct jobs, users must 
submit a job array. Each job within the array can have an arbitrary multinode,
multitask layout; the script will determine automatically what task it is 
running in the super-array of tasks.

It is the user's responsibility to ensure that the number of total tasks in the
job array times the delta-per-task covers all the blocks that need to be
computed. If there are extra tasks, they will not compute anything and waste
cluster resources. In either case, an appropriate warning is issued. 
"""
import numpy as np
from pspipe_utils import log, pspipe_list, covariance as psc
from pspy import so_dict, pspy_utils
from itertools import product
import os
import argparse
import warnings

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--delta-per-task', type=int, default=-1,
                    help='The number of blocks to compute in a given task. '
                    'If not a positive integer, then all the possible blocks')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

pspipe_ops_dir = d['pspipe_operators_dir']
covariances_dir = d['covariances_dir']
pspy_utils.create_directory(covariances_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

lmax = d['lmax']
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# format:
# - unroll all 'fields' i.e. (survey x array x chan x pol) is a 'field'
# - any given combination is then ('field'x'field' X 'field'x'field x 'spintype')
# - canonical spintypes are ('00', '02', '++', '--')
#
# notes:
# - we are 'hardcoding' that all splits for a given field have the same
# analysis mask, and that all pols for a given field have the same 
# sigma map.
# - we are taking advantage of fact that we have a narrow mapping of pol
# to spintypes.

# we define the canon by the windows order. we first build the fields,
# then use a mapping from fields to windows to build the canonical
# windows
sv_ar_chans = [] # necessary for indexing signal model
coadd_infos = [] # no splits, can't think of a better name
for sv1 in sv2arrs2chans:
    for ar1 in sv2arrs2chans[sv1]:
        for chan1 in sv2arrs2chans[sv1][ar1]:
            sv_ar_chans.append((sv1, ar1, chan1)) 
            for split1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
                for pol1 in ('T', 'E', 'B'):                    
                    coadd_info = (sv1, ar1, chan1, pol1)
                    if coadd_info not in coadd_infos:
                        coadd_infos.append(coadd_info)
                    else:
                        pass # coadd_infos are not unique because of splits

canonized_combos = {}

for sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4 in product(sv_ar_chans, repeat=4):

    # canonize the coadded fields
    sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq = psc.canonize_disconnected_4pt(
        sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4, sv_ar_chans
    )

    if (sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq) not in canonized_combos:
        canonized_combos[(sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq)] = [(sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4)]
    else:
        canonized_combos[(sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq)].append((sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4))

np.save(f'{covariances_dir}/canonized_split_averaged_binned_spec_cov_combos.npy', canonized_combos)

# get the indices of the blocks that will be computed in this job
delta_per_task = args.delta_per_task
if delta_per_task < 1:
    start = 0
    stop = len(canonized_combos)
else:
    job_array_idx = os.environ.get('SLURM_ARRAY_TASK_ID', 0)
    njob_array_idxs = os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)
    job_task_idx = os.environ.get('SLURM_PROCID', 0)
    njob_task_idxs = os.environ.get('SLURM_NPROCS', 1)

    total_tasks = njob_array_idxs * njob_task_idxs * delta_per_task
    if total_tasks < len(canonized_combos):
        warnings.warn(f'The total number of blocks computed across all tasks is {total_tasks} '
                        f'but the total number of blocks that need to be computed is '
                        f'{len(canonized_combos)}. {len(canonized_combos) - total_tasks} '
                        f'blocks will not be computed')
    elif total_tasks > len(canonized_combos):
        warnings.warn(f'The total number of blocks computed across all tasks is {total_tasks} '
                        f'but the total number of blocks that need to be computed is '
                        f'{len(canonized_combos)}. {total_tasks - len(canonized_combos)} '
                        f'tasks will not perform any computation and will waste resources')
    
    start = (njob_task_idxs * job_array_idx + job_task_idx) * delta_per_task
    stop = (njob_task_idxs * job_array_idx + job_task_idx + 1) * delta_per_task

    start = min(start, len(canonized_combos))
    stop = min(stop, len(canonized_combos))
    
    log.info(f'Computing only the blocks: {start}:{stop} of {len(canonized_combos)}')

# # main loop, we will stitch all pols together here
# # iterate over all pairs/orders of channels
for i in range(start, stop):
    (sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq) = list(canonized_combos.keys())[i]

    spec_cov_fn = f"{covariances_dir}/analytic_cov_{'_'.join(sv_ar_chani)}x{'_'.join(sv_ar_chanj)}"
    spec_cov_fn += f"_{'_'.join(sv_ar_chanp)}x{'_'.join(sv_ar_chanq)}.npy"

    if os.path.isfile(spec_cov_fn):
        log.info(f'{spec_cov_fn} exists, skipping')
    else:
        log.info(f'Generating {spec_cov_fn}')

        # binned pseudo_cov is always saved as double because it is inverted
        pseudo_cov = np.zeros((len(spectra) * (lmax - 2), len(spectra) * (lmax - 2)), dtype=np.float64)

        for ridx, (poli, polj) in enumerate(spectra):
            for cidx, (polp, polq) in enumerate(spectra):
                coadd_infoi = (*sv_ar_chani, poli)
                coadd_infoj = (*sv_ar_chanj, polj)
                coadd_infop = (*sv_ar_chanp, polp)
                coadd_infoq = (*sv_ar_chanq, polq)

                coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq = psc.canonize_disconnected_4pt(
                    coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq, coadd_infos
                    )

                pseudo_cov_fn = f"{covariances_dir}/pseudo_cov_{'_'.join(coadd_infoi)}x{'_'.join(coadd_infoj)}"
                pseudo_cov_fn += f"_{'_'.join(coadd_infop)}x{'_'.join(coadd_infoq)}.npy"

                log.info(f'Loading {pseudo_cov_fn}')
                pseudo_cov[ridx*(lmax - 2):(ridx+1)*(lmax - 2), cidx*(lmax - 2):(cidx+1)*(lmax - 2)] = \
                    np.load(pseudo_cov_fn)[2:lmax, 2:lmax] # NOTE: assumes 2:lmax ordering in spectra!

        Fij_fn = f"{pspipe_ops_dir}/Finv_Pbl_Minv_{'_'.join(sv_ar_chani)}x{'_'.join(sv_ar_chanj)}.npy"
        log.info(f'Loading {Fij_fn}')
        Fij = np.load(Fij_fn)

        Fpq_fn = f"{pspipe_ops_dir}/Finv_Pbl_Minv_{'_'.join(sv_ar_chanp)}x{'_'.join(sv_ar_chanq)}.npy"
        log.info(f'Loading {Fpq_fn}')
        Fpq = np.load(Fpq_fn)

        spec_cov = Fij @ pseudo_cov @ Fpq.T
        np.save(spec_cov_fn, spec_cov)