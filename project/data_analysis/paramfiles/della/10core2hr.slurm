#! /bin/bash
#
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH -t 2:00:00               # QOS is determined by time
#SBATCH -p physics
#SBATCH --output=slurmoutput/R-%j.out

cd /tigress/zequnl/PSpipe/project/data_analysis
module load intel/2021.1 openmpi/intel-2021.1/4.1.0 anaconda3/2023.3
export MPICC=$(which mpicc)
conda activate ps 

export OMP_NUM_THREADS=10
export JULIA_NUM_THREADS=10

$1
