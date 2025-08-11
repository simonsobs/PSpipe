#! /bin/bash
#
#SBATCH --account=simonsobs
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zatkins@princeton.edu
#SBATCH --output=/home/zatkins/projects/PSpipe_dev/slurm_output/slurm-%j.out

### set environment ###
module purge
module load tiger3/250723

# activate venv
source /home/zatkins/projects/PSpipe_dev/.venv/bin/activate

export SLURM_WAIT=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$@"