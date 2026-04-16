#! /bin/bash
#
#SBATCH --account=simonsobs
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=giardiellos@cardiff.ac.uk
#SBATCH --output=BASE_DIR/slurm_output/slurm-%j.out

### set environment ###
module purge
module load tiger3/250723

# activate venv
source BASE_DIR/.venv/bin/activate

export SLURM_WAIT=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$@"
