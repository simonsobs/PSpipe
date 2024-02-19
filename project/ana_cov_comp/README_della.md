

On Della, it's better to submit SLURM jobs via script instead of interactively. For 
convenience, let's define a commonly used job script,

```bash
export sbatch1pn="/home/zatkins/repos/simonsobs/PSpipe/project/ana_cov_comp/paramfiles/1physicsnode.slurm"
```

We need windows. Note, this will generate "PSpipe" style windows. A particular dict file may load other, pre-generated windows.

```bash
for i in {0..5}; do \
    sbatch --cpus-per-task 1 --mem 48G --time 20:00 --job-name get_window_dr6 ${sbatch1pn} "python -u python/get_window_dr6.py paramfiles/global_dr6_v4.dict $i $((i+1))"
done
```

We need to generate some signal spectra for the covariance matrix,

```bash
sbatch --cpus-per-task 1 --mem 2G --time 2:00 --job-name get_best_fit_mflike ${sbatch1pn} "python -u python/get_best_fit_mflike.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict"
```

Next we'll need to compute the mode-coupling matrices,

```bash
for i in {0..20}; do \
    sbatch --cpus-per-task 10 --mem 28G --time 5:00 --job-name get_mcm_and_bbl ${sbatch1pn} "python -u python/get_mcm_and_bbl.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict $i $((i+1))"
done
```

Next, we compute the alms of the maps.

```bash
sbatch --cpus-per-task 10 --mem 28G --time 20:00 --job-name get_alms ${sbatch1pn} "python -u python/get_alms.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict"
```

Now spectra,

```bash
sbatch --cpus-per-task 1 --mem 28G --time 10:00 --job-name get_spectra_from_alms ${sbatch1pn} "python -u python/get_spectra_from_alms.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict"
```

```bash
sbatch --cpus-per-task 1 --mem 1G --time 2:00 --job-name get_noise_model ${sbatch1pn} "python -u python/get_noise_model.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict"
```

```bash
sbatch --cpus-per-task 1 --mem 1G --time 2:00 --job-name get_per_split_noise ${sbatch1pn} "python -u python/get_per_split_noise.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict"
```

Now we compute the analytic covariances blocks assuming an isotropic noise model.

```bash
sbatch --cpus-per-task 4 --mem 10G --time 15:00 --job-name get_sq_windows_alms ${sbatch1pn} "python -u python/get_sq_windows_alms.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict"
```

```bash
for i in {0..230}; do \
    sbatch --cpus-per-task 10 --mem 32G --time 5:00 --job-name get_covariance_blocks ${sbatch1pn} "python -u python/get_covariance_blocks.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict $i $((i+1))"
done
```

For anisotropic covariances,

```bash
for i in {0..230}; do \
    sbatch --cpus-per-task 4 --mem 10G --time 15:00 --job-name get_split_covariance_aniso ${sbatch1pn} "python -u python/get_split_covariance_aniso.py paramfiles/sims_dr6_v4_lmax5400_20230816.dict $i $((i+1))"
done
```
