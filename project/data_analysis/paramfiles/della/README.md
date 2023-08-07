

On Della, it's better to submit SLURM jobs via script instead of interactively. For 
convenience, let's define a commonly used job script,

```bash
alias 8core1hr="sbatch paramfiles/della/8core1hr.slurm $1"  # test QOS
alias 10core2hr="sbatch paramfiles/della/10core2hr.slurm $1"  # short QOS
```

We need windows,

```bash
for i in {0..4}; do \
    10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/get_window_dr6.py paramfiles/della/global_dr6_v4.dict $i $((i+1))"
done
```

We need to generate some signal spectra for the covariance matrix,

```bash
8core1hr "srun --ntasks 1 --cpus-per-task 8 --cpu-bind=cores python -u python/get_best_fit_mflike.py paramfiles/della/global_dr6_v4.dict"
```

Next we'll need to compute the mode-coupling matrices,

```bash
for i in {0..15}; do \
    10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/get_mcm_and_bbl.py paramfiles/della/global_dr6_v4.dict $i $((i+1))"
done
```

Next, we compute the alms of the maps.

```bash
10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/get_alms.py paramfiles/della/global_dr6_v4.dict"
```

Now spectra,

```bash
10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/get_spectra_from_alms.py paramfiles/della/global_dr6_v4.dict"
```

```bash
10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/split_nulls/get_per_split_noise.py paramfiles/della/global_dr6_v4.dict"
```

Now we compute the analytic covariances blocks assuming an isotropic noise model.

```bash
10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/get_sq_windows_alms.py paramfiles/della/global_dr6_v4.dict"
```

```bash
for i in {0..119}; do \
    10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/get_covariance_blocks.py paramfiles/della/global_dr6_v4.dict $i $((i+1))"
done
```

For anisotropic covariances,

```bash
10core2hr "srun --ntasks 1 --cpus-per-task 10 --cpu-bind=cores python -u python/get_split_covariance_aniso.py paramfiles/della/global_dr6_v4.dict 100 101"
```
