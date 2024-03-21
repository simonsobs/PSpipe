#! /bin/bash
#
#SBATCH --partition=physics

########## REQUIRED USER CHANGE ##########
# SET PSPIPE ENVIRONMENT IN THIS BLOCK

##########################################

########## REQUIRED USER CHANGE ##########
# CD TO PSPIPE INSTALLATION IN THIS BLOCK

##########################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$@"