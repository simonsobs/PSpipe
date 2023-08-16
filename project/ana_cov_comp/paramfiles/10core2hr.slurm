#! /bin/bash
#
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH -t 2:00:00               # QOS is determined by time
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
