#! /bin/bash
#
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --partition=physics
#SBATCH --output=slurmoutput/slurm-%A_%a.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zatkins@princeton.edu

cd /home/zatkins/repos/simonsobs/PSpipe/project/ana_cov_comp

### set environment ###
module purge
module load cmb-della8/220426

# activate conda
conda activate pspy-della8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    srun $1
elif [[ -z "$2" ]]; then
    srun $1 $SLURM_ARRAY_TASK_ID
else
    srun $1 $(($2*$SLURM_ARRAY_TASK_ID)) $(($2*$SLURM_ARRAY_TASK_ID + $2))
fi