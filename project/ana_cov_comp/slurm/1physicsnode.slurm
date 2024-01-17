#! /bin/bash
#
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --partition=physics
#SBATCH --output=slurmoutput/R-%j.out
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

srun $1
