#! /bin/bash
#SBATCH -A mp107
#SBATCH -C cpu
#SBATCH -n 1                # node count
#SBATCH -c 128
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --output=slurmoutput/R-%j.out

cd /pscratch/sd/x/xzackli/so/PSpipe_ana_cov_comp/project/ana_cov_comp
base_dir=/pscratch/sd/x/xzackli/so/envs/ps
pyenv_dir=${base_dir}/pyenv/perlmutter
export SOFIND_SYSTEM=perlmutter
export PYTHONPATH=$PYTHONPATH:${software_dir}
source ${pyenv_dir}/bin/activate

export OMP_NUM_THREADS=128

$1

