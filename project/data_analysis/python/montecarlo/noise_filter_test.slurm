#!/bin/bash
#SBATCH --qos=regular
#SBATCH --account=mp107
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=haswell

cd /global/cscratch1/sd/xzackli/PSpipe/project/data_analysis/

module load python intel
source activate ps38
export PYTHONPATH=/global/cscratch1/sd/xzackli/PSpipe/project/data_analysis/python/:/global/homes/x/xzackli/src/enlib:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=32

echo "param: $1, noisetype: $2, nametag: $3, filter_noise $4"

srun python -u python/montecarlo/mc_mnms_get_spectra.py $1 $5 $5 --noisetype $2 --nametag $3 --filter_noise $4
