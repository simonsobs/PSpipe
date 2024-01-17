#!/bin/bash
# 
#SBATCH --job-name=get_simulated_spectra_and_results     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zatkins@princeton.edu
#SBATCH --array=2               # job array with index values 0-nsplit-1

echo $(hostname)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### set environment ###
module purge
module load cmb-della8/220426

# activate conda
conda activate pspy-della8

scenario=${SLURM_ARRAY_TASK_ID}
num_sims=1000

python -u /home/zatkins/repos/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/get_simulated_spectra_and_results.py \
--scenario ${scenario} --num-sims ${num_sims}
