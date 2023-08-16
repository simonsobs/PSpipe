#! /bin/bash
#
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH -p physics
#SBATCH --output=slurmoutput/R-%j.out

cd /home/zatkins/repos/simonsobs/PSpipe/project/data_analysis

### set environment ###
module purge
module load cmb-della8/220426

# activate conda
conda activate pspy-della8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

$1
