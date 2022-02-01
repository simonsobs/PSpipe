#!/bin/bash
#SBATCH --qos=regular
#SBATCH --account=mp107
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell
#SBATCH --array=0-15

cd /global/cscratch1/sd/xzackli/PSpipe/project/data_analysis/

module load python intel
source activate ps38
export PYTHONPATH=/global/cscratch1/sd/xzackli/PSpipe/project/data_analysis/python/:/global/homes/x/xzackli/src/enlib:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=16

srun python -u python/montecarlo/mc_get_spectra.py paramfiles/global_dr6_v3_4pass_pa6_tiled.dict $SLURM_ARRAY_TASK_ID
