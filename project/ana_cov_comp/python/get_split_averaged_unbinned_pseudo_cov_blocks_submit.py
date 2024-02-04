"""
Given the grouping of blocks produced by
get_split_averaged_unbinned_pseudo_cov_blocks_recipe.py, this script submits
jobs for all groups of blocks to run 
get_split_averaged_unbinned_pseudo_cov_blocks.py.

This script is necessary because each group of block has different memory
requirements, set by the number of files loaded in each group, so a job array
is not really possible.
"""
from pspy import so_dict
from pspipe_utils import log 

import numpy as np

import os
import subprocess
import argparse

REL_MEM_SAFETY_FAC = 1.2
ABS_MEM_SAFETY_FAC_GB = 5

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', 
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('slurm_template',
                    help='Filename (full or relative path) of slurm submission script template to use')
parser.add_argument('-c', '--cpus-per-task', dest='cpus_per_task', type=int, default=1,
                    help='Number of cpus to allocate per job')
parser.add_argument('-t', '--time', dest='time', type=str, default='60:00',
                    help='Time limit per job; see sbatch documentation for acceptable formats')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

covariances_dir = d['covariances_dir']

lmax = d['lmax']
coupling_template = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
mem_per_coupling_gb = coupling_template.size * coupling_template.itemsize / 1e9

# our job is to:
# 1. get the list of jobs to submit
# - this is the list of groups of blocks
# 2. get the the number of couplings to be loaded 
# - this becomes the memory per job
# 3. submit each job

# load the recipe
recipe = np.load(f'{covariances_dir}/canonized_split_averaged_unbinned_pseudo_cov_blocks_recipe.npz')
ngroups = len(recipe.files) - 2 # 2 files are not group indices

for i in range(ngroups):
    num_couplings = recipe['group_total_reads'][i]
    mem_gb = int(np.ceil(REL_MEM_SAFETY_FAC * num_couplings * mem_per_coupling_gb + ABS_MEM_SAFETY_FAC_GB))
    cmds = f'sbatch --cpus-per-task {args.cpus_per_task} --mem {mem_gb}G --time {args.time}'.split(' ')
    cmds += f'--job-name get_split_averaged_unbinned_pseudo_cov_blocks {args.slurm_template}'.split(' ')
    cmds += [f"python -u {os.path.join(os.path.dirname(__file__), 'get_split_averaged_unbinned_pseudo_cov_blocks.py')} {args.paramfile} {i}"]
    subprocess.run(cmds)