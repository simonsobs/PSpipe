#! /bin/bash
#
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --partition=physics

########## REQUIRED USER CHANGE ##########
# SET PSPIPE ENVIRONMENT IN THIS BLOCK

##########################################

########## REQUIRED USER CHANGE ##########
# CD TO PSPIPE INSTALLATION IN THIS BLOCK

##########################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    srun $1
elif [[ -z "$2" ]]; then
    srun $1 $SLURM_ARRAY_TASK_ID
else
    srun $1 $(($2*$SLURM_ARRAY_TASK_ID)) $(($2*$SLURM_ARRAY_TASK_ID + $2))
fi