description = """
Given the grouping of blocks -- the "recipe" -- produced by
get_split_averaged_unbinned_pseudo_cov_blocks.py run in recipe mode, this
script submits jobs for all groups of blocks to run 
get_split_averaged_unbinned_pseudo_cov_blocks.py in compute mode. Thus, the
user should never run get_split_averaged_unbinned_pseudo_cov_blocks.py in
compute mode directly, they should only run this script. This script is necessary
because each group of blocks has different memory requirements, set by the number
of files loaded in each group, so a job array is not easily possible, and
submitting jobs by hand is tedious.

There is some subtlety to using this script, coupled to how a user's cluster
prioritizes and charges for jobs. This script will try to pack as many tasks
onto distinct single-node jobs as possible, subject to some user-provided 
limits. In particular, the user specifies the maximum gb a single job can
request to cover all its tasks, as well as the maximum cpus a single job can
request to covera all its tasks. If a user's cluster priority/charges optimize
for fewer jobs with more tightly packed tasks, one will want to specify the
maximum gb and cpus on a single node. If a users's cluster priority/charges
optimize for many jobs with few resources, and handles packing of jobs on nodes
on its own with no penalty, one will want to artifically lower (for example) the
maximum cpus a single job can request to the supplied --cpus-per-task, so that
each task is submitted in a distinct job requesting as few resources as possible.

Otherwise, --cpus-per-task needs to be specified directly to this script. Any other
slurm arguments should be passed together in quotation marks to --slurm-args, except
those relating to memory, cpus-per-task, ntasks, or the job name, since those are
handled automatically by this script.

It is highly recommended to run this script with --dry-run first to ensure that
all job submissions statistics and commands are as expected. Modifications can
be made to the recipe or to the arguments of the scipt to optimize for resources
required for each job, and job packing on nodes.
"""
from pspy import so_dict
from pspipe_utils import log 

import numpy as np

import os
import subprocess
import argparse

REL_MEM_SAFETY_FAC = 1.05
ABS_MEM_SAFETY_FAC_GB = 4

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', 
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--dry-run', default=False, action='store_true',
                    help='Instead of submitting all the jobs, print how jobs and tasks '
                    'will be allocated across nodes. Helpful to gauge if recipe or resources '
                    'should be adjusted to maximize node packing efficiency')
parser.add_argument('--slurm-template', type=str, required=True,
                    help='Filename (full or relative path) of slurm submission script template to use')
parser.add_argument('--max-mem-gb-per-single-node-job', type=int, required=True,
                    help='Maximum memory in GB that can be requested of a single-node job')
parser.add_argument('--max-cpus-per-single-node-job', type=int, required=True,
                    help='Maximum number of cpus that can be requested of a single-node job')
parser.add_argument('--cpus-per-task', type=int, default=1,
                    help='The cpus for each task. Must be supplied outside of slurm-args '
                    'to allow for script to automatically assign tasks to nodes.')
parser.add_argument('--slurm-args', type=str, default=None,
                    help='Any other arguments to pass to the sbatch command of each job '
                    '(same for all jobs). Entire set of arguments should be in quotation '
                    'marks. Supply arguments related to --cpus-per-task as an argument to '
                    'this script, not here! Also, arguments related to --ntasks, --job-name, and '
                    '--mem will be provided automatically by the script, so do not supply '
                    'them here! Examples of arguments that should go here are, e.g., '
                    '--time, --dependency, --constraint, --account, --partition etc.')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

covariances_dir = d['covariances_dir']

lmax_pseudocov = d['lmax_pseudocov']
assert lmax_pseudocov >= d['lmax'], \
    f"{lmax_pseudocov=} must be >= {d['lmax']=}" 
dtype_pseudocov = d['dtype_pseudocov']

mem_per_coupling_gb = (lmax_pseudocov + 1)**2 * np.dtype(dtype_pseudocov).itemsize / 1e9

# load the recipe
recipe = np.load(f'{covariances_dir}/canonized_split_averaged_unbinned_pseudo_cov_blocks_recipe.npz')
ngroups = len(recipe.files) - 2 # 2 files are not group indices
group_total_reads = recipe['group_total_reads']
assert ngroups == len(group_total_reads), \
    f'{ngroups=} does not equal {len(group_total_reads)=}'

# we will try to pack multiple groups onto single nodes. to figure out 
# which groups will go on which nodes, we simply iterate in order of
# the groups and greedily add them to a node until the next group's
# memory or cpu requirements would exceed the limits specified in the
# arguments to this script. once that happens, we package those groups
# and move onto the next node
unassigned_groups = np.arange(ngroups)
jobs2groups = []
while len(unassigned_groups) > 0:
    groups_in_job = []
    assigned_mem_in_job = 0
    assigned_cpus_in_job = 0
    
    # simple algorithm: just append groups to this job in order of groups
    # until the mem or cpus would go beyond the limit
    for group in unassigned_groups:
        num_couplings_in_group = group_total_reads[group]
        mem_in_group_gb = int(np.ceil(REL_MEM_SAFETY_FAC * mem_per_coupling_gb * num_couplings_in_group + ABS_MEM_SAFETY_FAC_GB))
        cpus_in_group = args.cpus_per_task

        if (assigned_mem_in_job + mem_in_group_gb <= args.max_mem_gb_per_single_node_job) and (assigned_cpus_in_job + cpus_in_group <= args.max_cpus_per_single_node_job):
            groups_in_job.append(group)
            assigned_mem_in_job += mem_in_group_gb
            assigned_cpus_in_job += cpus_in_group
        else:
            break

    assert len(groups_in_job) > 0, \
        f'{group=} cannot fit on a single-node even if it were the only task on that node!'

    jobs2groups.append(groups_in_job)    

    # because we went in order of unassigned_groups in the for loop, 
    # the indexes of unassigned groups to be deleted is just the first
    # len(groups_in_job) in unassigned groups
    unassigned_groups = np.delete(unassigned_groups, np.arange(len(groups_in_job)))

# submit each single-node job
total_ntasks = 0
total_mem_gb = 0
total_cpus = 0
all_cmd_strs = []
for i, groups in enumerate(jobs2groups):
    ntasks = len(groups)
    mem_in_job_gb = 0
    for group in groups:
        num_couplings_in_group = group_total_reads[group]
        mem_in_job_gb += int(np.ceil(REL_MEM_SAFETY_FAC * mem_per_coupling_gb * num_couplings_in_group + ABS_MEM_SAFETY_FAC_GB))
    cpus_in_job = ntasks * args.cpus_per_task

    total_ntasks += ntasks
    total_mem_gb += mem_in_job_gb
    total_cpus += cpus_in_job

    cmds = f'sbatch -N 1 --mem {mem_in_job_gb}G --ntasks {len(groups)} --cpus-per-task {args.cpus_per_task}'.split()
    cmds += f'--job-name get_split_averaged_unbinned_pseudo_cov_blocks'.split()
    if args.slurm_args:
        cmds += args.slurm_args.split()
    cmds += f'{args.slurm_template}'.split()
    cmds += f"python -u {os.path.join(os.path.dirname(__file__), 'get_split_averaged_unbinned_pseudo_cov_blocks.py')}".split()
    cmds += f"{args.paramfile} --group-idxs {' '.join(str(g) for g in groups)}".split()
    
    if args.dry_run:
        print(f'Job {i}: ntasks={ntasks}, mem={mem_in_job_gb}G, cpus={cpus_in_job}')
        all_cmd_strs.append(' '.join(cmds))
    else:
        subprocess.run(cmds)

if args.dry_run:
    print(f'Totals: jobs={len(jobs2groups)}, tasks={total_ntasks}, mem={total_mem_gb}G, cpus={total_cpus}', end='\n\n')
    for i, cmd_str in enumerate(all_cmd_strs):
        print(f'Job {i}: {cmd_str}')