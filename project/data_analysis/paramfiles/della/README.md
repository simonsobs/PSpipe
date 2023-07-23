

On Della, it's better to submit SLURM jobs via script instead of interactively. For 
convenience, let's define a commonly used job script,

```bash
alias 8core1hr="sbatch paramfiles/della/8core1hr.slurm $1"  # test QOS
alias 8core2hr="sbatch paramfiles/della/8core2hr.slurm $1"  # short QOS
```

We need windows,

```bash
for i in {0..4}; do \
    8core2hr "srun --ntasks 1 --cpus-per-task 8 --cpu-bind=cores python -u python/get_window_dr6.py paramfiles/della/global_dr6_v4.dict $i $((i+1))"
done
```

We need to generate some signal spectra for the covariance matrix,

```bash
8core1hr "srun --ntasks 1 --cpus-per-task 8 --cpu-bind=cores python -u python/get_best_fit_mflike.py paramfiles/della/global_dr6_v4.dict"
```

Next we'll need to compute the mode-coupling matrices,

```bash
for i in {0..15}; do \
    8core2hr "srun --ntasks 1 --cpus-per-task 8 --cpu-bind=cores python -u python/get_mcm_and_bbl.py paramfiles/della/global_dr6_v4.dict $i $((i+1))"
done
```
