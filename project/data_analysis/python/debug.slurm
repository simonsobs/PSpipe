#!/bin/bash
#SBATCH --qos=debug
#SBATCH --account=mp107
#SBATCH --time=0:30:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=haswell

cd /global/cscratch1/sd/xzackli/PSpipe/project/data_analysis/

module load python intel
source activate ps38
export PYTHONPATH=/global/cscratch1/sd/xzackli/PSpipe/project/data_analysis/python/:/global/homes/x/xzackli/src/enlib:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=32

echo "noisetype: $1"

srun -n 64 -c 64 --cpu_bind=cores python python/montecarlo/mc_get_kspace_tf_spectra.py paramfiles/global_dr6_v3_4pass_pa6_filt_test.dict

