#! /bin/bash
#
#SBATCH -p physics
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH -t 1:00:00


module load anaconda3
PATH=/home/zequnl/src/julia-1.6.0/bin/:$PATH; export PATH
export OMP_NUM_THREADS=4
export JULIA_NUM_THREADS=4

$1

