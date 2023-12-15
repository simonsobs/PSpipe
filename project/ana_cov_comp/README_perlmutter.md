
## Window alms:
```bash
alias shortjob="sbatch paramfiles/1perlmutternode.slurm $1"  # short QOS
shortjob "srun --ntasks 1 --cpus-per-task 128 --cpu-bind=cores python -u \
python/get_1pt_ewin_alms.py paramfiles/cov_dr6_v4_20231128.dict"
shortjob "srun --ntasks 1 --cpus-per-task 128 --cpu-bind=cores python -u \
python/get_2pt_ewin_alms.py paramfiles/cov_dr6_v4_20231128.dict"
```

## Couplings:
2pt couplings:
```bash
alias shortjob="sbatch --time 2:00:00 paramfiles/1perlmutternode.slurm $1"  # short QOS
shortjob "srun --ntasks 1 --cpus-per-task 128 --cpu-bind=cores python -u \
python/get_2pt_coupling_matrices.py paramfiles/cov_dr6_v4_20231128.dict"
```

For production on all 4pt,
```bash
alias shortjob="sbatch paramfiles/1perlmutternode.slurm $1"  # short QOS
for i in {0..57}; do \
    shortjob "srun --ntasks 1 --cpus-per-task 128 --cpu-bind=cores python -u \
        python/get_4pt_coupling_result.py paramfiles/cov_dr6_v4_20231128.dict $((50*i)) $((50*i+50))"
done
```

## Pseudosignal Matrix

First, generate the bestfit foregrounds as usual,
```bash
python python/get_best_fit_mflike.py paramfiles/cov_dr6_v4_20231128.dict 
```

Then generate the pseudodiagonal matrix,
```bash
python python/get_pseudosignal.py paramfiles/cov_dr6_v4_20231128.dict
```
