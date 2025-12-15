For running the dr6xdeep56_20251019 or dr6xdeep56_20251119 pipeline. This has: analytic kspace filter,
no beam error, leakage, or leakage error, and only cosmic variance and map
noise in the covariance (TODO: add these pieces, fg_covs, lensing).

# Setup
1. Install `PSpipe` dependencies in an environment
2. Download `PSpipe` to a directory on your system
3. Make a directory on your system for `PSpipe` input and output. It can be
named whatever you want, but we'll refer to it as the `data_dir` 
5. Locate all required pre-existing products on your system. Either copy or 
symlink above products into subdirectories with those names inside the
`data_dir`. It should look like this:
    ```bash
    /path/to/my/PSpipe/data_dir
    ├── beams
    │   ├── dr6
    │   │   ├── leakage_beams
    │   │   │   ├── pa4_f220_gamma_t2b.txt
    │   │   │   ├── pa4_f220_gamma_t2e.txt
    │   │   │   ├── pa5_f090_gamma_t2b.txt
    │   │   │   ├── pa5_f090_gamma_t2e.txt
    │   │   │   ...
    │   │   │   ├── pa6_f150_gamma_t2b.txt
    │   │   │   └── pa6_f150_gamma_t2e.txt
    │   │   └── main_beams
    │   │       ├── coadd_pa4_f220_night_beam_tform_jitter_cmb.txt
    │   │       ├── coadd_pa5_f090_night_beam_tform_jitter_cmb.txt
    │   │       ...
    │   │       └── coadd_pa6_f150_night_beam_tform_jitter_cmb.txt
    │   └── lat
    │       └── iso
    │           ├── leakage_beams
    │           └── main_beams
    ├── binning
    │   └── binning_50.dat
    ├── maps
    │   ├── dr6
    │   │   ├── deep56
    │   │   └── published -> /scratch/gpfs/ACT/data/act_dr6/dr6.02/maps/published
    │   └── lat
    │       └── deep56
    │           ├── 20251019
    │           │   ├── sky_ivar_c1_f220_coadd_20251019.fits
    │           │   ├── sky_ivar_c1_f220_set0_20251019.fits
    │           │   ...
    │           │   ├── sky_ivar_i6_f150_set3_20251019.fits
    │           │   ├── sky_map_c1_f220_coadd_20251019.fits
    │           │   ├── sky_map_c1_f220_set0_20251019.fits
    │           │   ...
    │           │   └── sky_map_i6_f150_set3_20251019.fits
    │           └── 20251119
    │               ├── sky_ivar_c1_f220_coadd_20251119.fits
    │               ├── sky_ivar_c1_f220_set0_20251119.fits
    │               ...
    │               ├── sky_ivar_i6_f150_set3_20251119.fits
    │               ├── sky_map_c1_f220_coadd_20251119.fits
    │               ├── sky_map_c1_f220_set0_20251119.fits
    │               ...
    │               └── sky_map_i6_f150_set3_20251119.fits
    ├── mask
    │   ├── HFI_Mask_GalPlane-apo0_2048_R2.00_GAL070_fejer1.fits
    │   └── source_mask_15mJy_and_dust_rad12.fits
    └── passbands
        ├── dr6
        │   ├── passband_dr6_pa4_f220.dat
        │   ├── passband_dr6_pa5_f090.dat
        │   ...
        │   └── passband_dr6_pa6_f150.dat
        └── lat
            └── iso
                ├── bandpass_mean_f090.dat
                ├── bandpass_mean_f150.dat
                ├── bandpass_mean_f220.dat
                └── bandpass_mean_f280.dat
    ```

Notes:
* The `dr6` products are all public, including the masks. For convenience, the 
above user has symlinked to a public directory on the `tiger` cluster holding the
`dr6` maps, rather than copy them.
* The `lat_iso` maps have been distributed by Carlos
* The mean bandpass files have been produced by Merry using
 `/path/to/PSpipe/project/SO/pISO/python/passbands/bandpasses_format.py`, so you can ask 
him for these files. 
4. Modify the paramfile (e.g.,
 `/path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`) with the `data_dir`
from step 3, and a `yaml_dir` containing auxiliary `PSpipe` configuration files.
Default ones are distributed in the code at 
 `/path/to/PSpipe/project/SO/pISO/python/yaml`.
5. (Optional) The slurm submission for scripts can be done however you like, but
we demonstrate one way below, using an sbatch script `/path/to/sbatch/script.slurm`,
which sets up standard sbatch commands for the user, and executes whatever 
comes afterwards, e.g.
    ```bash
    #! /bin/bash
    #
    #SBATCH --account=simonsobs
    #SBATCH --mail-type=begin        # send email when job begins
    #SBATCH --mail-type=end          # send email when job ends
    #SBATCH --mail-user=zatkins@princeton.edu
    #SBATCH --output=/home/zatkins/projects/PSpipe_dev/slurm_output/slurm-%j.out

    ### set environment ###
    module purge
    module load tiger3/250723

    # activate venv
    source /home/zatkins/projects/PSpipe_dev/.venv/bin/activate

    export SLURM_WAIT=0
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

    "$@"
    ```

# DR6/LAT Preprocessing 
We need a few products to run the actual pipeline. These are very fast to 
produce and only need to be run once at the beginning (we would never recommend
this, but you could run these steps on a headnode):
1. Extract the `dr6` maps into the `deep56` geometry:
    - command: `python -u /path/to/PSpipe/project/SO/pISO/python/act/extract_act.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
2. Make some fiducial Gaussian beams with no beam error:
    - command: `python -u /path/to/PSpipe/project/SO/pISO/python/beams/get_gaussian_beams.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
3. Get the `extra_mask` for each map. These are the unapodized boolean footprints
that cut out map edges and noisy regions. For the `deep56` region, the `ivar`-based
cuts in the script are nonsensical for `dr6`, so we only iterate over the 
`lat-iso` maps. This is specified via `xtra_mask_yaml = yaml_dir + 'xtra_mask.yaml'`
in the paramfile, and 
    ```yaml
    surveys_to_xtra_mask:
        - lat_iso
    ```
    in the yaml file.
    - command: `sbatch --mem 48G --cpus-per-task 4 --time 10:00 --job-name get_xtra_mask /path/to/sbatch/script.slurm python -u /path/to/PSpipe/project/SO/pISO/python/masks/get_xtra_mask.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
4. Get the source subtracted LAT maps. for that we will use the ACT maps and source subtracted maps to get a source map and subtract it from LAT maps. This is meant to be a temporary solution before we get actual source subtracted maps.
    - command : `python /path/to/PSpipe/project/SO/pISO/python/act/subtract_source.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
  


After this point, the `data_dir` should look like:
```bash
/path/to/my/PSpipe/data_dir
├── beams
│   ├── dr6
│   │   ├── leakage_beams
│   │   │   ├── pa4_f220_gamma_t2b.txt
│   │   │   ...
│   │   │   └── pa6_f150_gamma_t2e.txt
│   │   └── main_beams
│   │       ├── coadd_pa4_f220_night_beam_tform_jitter_cmb.txt
│   │       ...
│   │       └── coadd_pa6_f150_night_beam_tform_jitter_cmb.txt
│   └── lat
│       └── iso
│           ├── leakage_beams
│           └── main_beams
│               ├── beam_gaussian_fwhm_0.8_arcmin_no_error.txt
│               ...
│               └── beam_gaussian_fwhm_7.4_arcmin_no_error.txt
├── binning
│   └── binning_50.dat
├── maps
│   ├── dr6
│   │   ├── deep56
│   │   │   ├── act_dr6.02_std_AA_night_pa4_f220_4way_coadd_ivar.fits
│   │   │   ...
│   │   │   └── act_dr6.02_std_AA_night_pa6_f150_4way_set3_map_srcfree.fits
│   │   └── published -> /scratch/gpfs/ACT/data/act_dr6/dr6.02/maps/published
│   └── lat
│       └── deep56
│           ├── 20251019
│           │   ├── sky_ivar_c1_f220_coadd_20251019.fits
│           │   ...
│           │   ├── sky_ivar_i6_f150_set3_20251019.fits
│           │   ├── sky_map_c1_f220_coadd_20251019.fits
│           │   ...
│           │   └── sky_map_i6_f150_set3_20251019.fits
│           └── 20251119
│               ├── sky_ivar_c1_f220_coadd_20251119.fits
│               ...
│               ├── sky_ivar_i6_f150_set3_20251119.fits
│               ├── sky_map_c1_f220_coadd_20251119.fits
│               ...
│               └── sky_map_i6_f150_set3_20251119.fits
├── mask
│   ├── dr6xdeep56_20251119
│   │   ├── xtra_mask_intersect.fits
│   │   ├── xtra_mask_lat_iso_c1_f220.fits
│   │   ├── xtra_mask_lat_iso_c1_f220_set0.fits
│   │   ...
│   │   ├── xtra_mask_lat_iso_i6_f150_set3.fits
│   │   └── xtra_mask_union.fits
│   ├── HFI_Mask_GalPlane-apo0_2048_R2.00_GAL070_fejer1.fits
│   └── source_mask_15mJy_and_dust_rad12.fits
├── passbands
│   ├── dr6
│   │   ├── passband_dr6_pa4_f220.dat
│   │   ...
│   │   └── passband_dr6_pa6_f150.dat
│   └── lat
│       └── iso
│           ├── bandpass_mean_f090.dat
│           ├── bandpass_mean_f150.dat
│           ├── bandpass_mean_f220.dat
│           └── bandpass_mean_f280.dat
└── plots
    └── dr6xdeep56_20251119
        └── mask
            ├── maps_ivar
            │   ├── sky_ivar_c1_f220_set0_20251119.png
            │   ...
            │   └── sky_map_i6_f150_set3_20251119_2.png
            ├── xtra_mask_intersect.png
            ├── xtra_mask_lat_iso_c1_f220.png
            ├── xtra_mask_lat_iso_c1_f220_set0.png
            ...
            ├── xtra_mask_lat_iso_i6_f150_set3.png
            └── xtra_mask_union.png
```

Notes:
* For `dr6`, the paramfile uses the produced `xtra_mask_union.fits` which includes
all `lat_iso` data. Thus each `dr6` map will have the same mask and windows.
* The products that have been produced are stored in a directory scheme like:
    ```bash
    {product_name}/{run_name}
    ```
    where `{product_name}` is, e.g., `mask` or `plots`, and `{run_name}` is
    specified in tha paramfile.
* All paths in paramfiles evaluate to full paths. Users can exploit this directory
scheme to run a modified pipeline that, e.g., uses some products from a different
run.

# Planck Preprocessing

Here are some specific instructions to pre-process Planck maps, namely to project them and to subtract the planck bright sources.
(A large part of this section comes from DR6xPlanck)

## Extract products

### Beams
Running `/path/to/PSpipe/project/SO/pISO/python/planck/extract_planck_beams.py` will extract and plot planck's beams in the right folder. You just need to specify where the original beams stand with `planck_fits_beam_path`. 
It also extract some "extended" NPIPE beams and save these in both legacy and NPIPE folders, these will be used for source subtraction.
```bash
python {python_path}/extract_planck_beams.py {paramfile}
```

### Maps projection
`/path/to/PSpipe/project/SO/pISO/python/planck/project_planck_maps.py` will project planck maps and ivar on the patch specified by the map at `planck_projection_template` (you can use an already projected ACT or LAT map for instance).
TODO : project planck masks ?
```bash
salloc -N 1 -C cpu -q interactive -t 01:00:00
OMP_NUM_THREADS=32 srun -n 8 -c 32 --cpu_bind=cores python {python_path}/project_planck_maps.py {paramfile}
```
This script can run in about 10 minutes (depends on the template size).

### Passbands and other
`planck_symlinks.sh` creates the right symlinks for passbands and other (?).
```bash
bash {python_path}/planck_symlinks.sh {paramfile}
```

## Subtract point-sources
You first need to extract the source catalog defined by `planck_source_catalog` in the paramfile with `/path/to/PSpipe/project/SO/pISO/python/planck/reformat_source_catalog.py`. You can then run the source subtraction using the 2 bash file. Note that you need to specify the path of your dory file with `dory_path` (you can use `/path/to/PSpipe/project/SO/pISO/python/planck/dory.py`, you just need to install enlib). These scripts read maps at `maps_dir_planck/{npipe|legacy}/` and make _srcfree maps.
You need to run this part with an interactive allocation, it takes around 10 minutes per map :
```bash
salloc -N 1 -C cpu -q interactive -t 03:00:00
bash {python_path}/run_legacy_src_subtraction_interactive.sh {legacy_paramfile}
bash {python_path}/run_npipe_src_subtraction_interactive.sh {npipe_paramfile}
```

## Compute spectra

You can then add "planck" to `surveys`, and all associated products in a paramfile. Please note that Planck beam only go up to ell = 4000, so you need `lmax` lower than this.
Planck can be included in spectra computation for calib or transfer function estimations.

## Leakage corrections

We can correct the power spectra from  T->P beam leakage using the script `/path/to/PSpipe/project/SO/pISO/python/leakage/get_leakage_corrected_spectra_per_split.py`. This subtracts from each data spectra the expected contribution from the leakage computed from the planet beam leakage measurement and a best fit model. Note that the correction due to not exactly knowing the best-fit model is going to be a second order one. The corrected spectra are saved in the directory `spectra_leak_corr_dir`, whose path is read from the parameter file. 
We then use `/path/to/PSpipe/project/SO/pISO/python/leakage/get_leakage_sim` to compute the montecarlo simulations needed to estimate the covariance of the spectra due to the uncertainties in the leakage model. The simulations are saved at the path `montecarlo_beam_leakage_dir`. The covariance is computed with `/path/to/PSpipe/project/SO/pISO/python/leakage/get_leakage_dir`, saving the leakage contribution to the total covariance in the covariance directory `cov_dir`.

You can run these scripts with:
```bash
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu

OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python /path/to/PSpipe/project/SO/pISO/python/leakage/get_leakage_corrected_spectra_per_split.py {paramfile}

OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python /path/to/PSpipe/project/SO/pISO/python/leakage/get_leakage_sim.py {paramfile}

OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python /path/to/PSpipe/project/SO/pISO/python/leakage/get_leakage_covariance.py {paramfile}
```

The same pipeline can also be used to compute ACT DR6 beam leakage corrected spectra and covariance.

## End-to-end sim correction

To start the generation of Planck montecarlo simulations, we start with the `path/to/PSpipe/project/SO/pISO/python/planck/get_planck_sim_nlms` script, used to get the noise alms for Planck NPIPE or legacy noise simulations. These noise maps are used to generate simulations of CMB + foregrounds + noise in the `path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_mnms_get_spectra_from_nlms.py` script, saved at the path `sim_spec_dir`.

You can run these using:
```bash
salloc -N 4 -C cpu -q interactive -t 02:00:00

OMP_NUM_THREADS=4 srun -n 256 -c 4 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/planck/get_planck_sim_nlms.py {paramfile}

OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_mnms_get_spectra_from_nlms_per_split.py {paramfile}
```

To compute the montecarlo contribution to the covariance matrix, we run `path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_analysis` (computing the average and standard deviation of the sims), `path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_cov_analysis.py` (computing the montecarlo contribution to the covariance) and plotting functions:
```bash
salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_analysis.py {paramfile}
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_cov_analysis.py {paramfile}
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_plot_spectra.py {paramfile}
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_plot_covariances.py {paramfile}
```

We can also generate noise-only simulations with `path/to/PSpipe/project/SO/pISO/python/planck/get_planck_spectra_correction_from_nlms.py`, to compute the correlated residual measured in the AxB NPIPE simulations or hm1xhm2 legacy simulations. These spectra are saved in the `sim_spectra_planck_noise_and_syst_dir` path.
We then compute the mean and standard deviation of these simulations with `path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_analysis`, using the flag `--planck-correction`, to point the code to the `sim_spectra_planck_noise_and_syst_dir` directory. The average of these corrections are saved at the `planck_mc_correction_dir` path. Finally, the Planck spectra (already corrected by the leakage and read from the `spectra_leak_corr_dir` directory), are also corrected for the correlated residuals from the end-to-end simulations running `path/to/PSpipe/project/SO/pISO/python/planck/get_corrected_planck_spectra.py`. The final spectra are saved in `spectra_leak_corr_planck_bias_corr_dir`.

You can run it with:
```bash
salloc -N 4 -C cpu -q interactive -t 03:00:00
OMP_NUM_THREADS=32 srun -n 32 -c 32 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/planck/get_planck_spectra_correction_from_nlms.py {paramfile}

salloc -N 1 -C cpu -q interactive -t 01:00:00
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/montecarlo/mc_analysis.py --planck-correction {paramfile}

salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python path/to/PSpipe/project/SO/pISO/python/planck/get_corrected_planck_spectra.py {paramfile}
```

## Dust-in-patch
We need to compute how much dust is in the patch we are using, in order to have priors on the dust amplitudes in temperature and polarization to use in the MCMC runs. We can estimate how much dust there is by fitting the residual dust in the difference of the Planck 353 and 143 maps. When using the temperature maps, we also fit for the CIB. 

To run the script to fit for the dust use:
```bash
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu

OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python path/to/PSpipe/project/SO/pISO/python/dust/fit_dust_amplitude.py {paramfile} --mode TT
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python path/to/PSpipe/project/SO/pISO/python/dust/fit_dust_amplitude.py {paramfile} --mode TE
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python path/to/PSpipe/project/SO/pISO/python/dust/fit_dust_amplitude.py {paramfile} --mode TB
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python path/to/PSpipe/project/SO/pISO/python/dust/fit_dust_amplitude.py {paramfile} --mode EE
OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python path/to/PSpipe/project/SO/pISO/python/dust/fit_dust_amplitude.py {paramfile} --mode BB
```

The `path/to/PSpipe/project/SO/pISO/python/dust/fit_dust_amplitude.py` can be run with additional flags in case we want to use the leakage and end-to-end sim corrected spectra (`--leak-corr`), the ACT DR6 220 channel for the CIB fit (`--use-220` and `--dr6-result-path-spectra` + `--dr6-result-path-covariance`), in case we want to sampled over the beta of CIB (--sample_beta) or set them to a value different than the default 2.20 (--beta_value), which is the value preferred by Planck data.

## Contribution to the covariance matrix from foreground parameters uncertainty

When doing null tests (which are done assuming a best-fit model for the CMB and the foregrounds), it may be useful to fold our uncertainty on the foreground parameters in the covariance. Especially TT null tests may improve significantly with the increase in the error bars due to the foreground marginalization. The marginalization is simply done at the covariance matrix level, considering just the uncertainty on the foregorund amplitudes (the theory model needs to be linear with respect to the parameters we marginalize over at the covmat level, and only the amplitudes easily satisfy this condition). So far, we use the uncertainty on these parameters from the ACT DR6 chains (read from `p_act_chain_filename`). If we want to read the uncertainty from an external covmat, point to its position using the `fg_covmat_path` in the parameter file. 
To compute this extra contribution to the covmat, run `get_fg_covariance_blocks.py`. It will produce blocks with name `fg_marginalization_cov_...`.

This applies not only to Planck but to any null test.


In the end, you should have the following added to your `data_dir` :
```bash
/path/to/my/PSpipe/data_dir
├── alms
│   └── planck
│       ├── alms_Planck_f100.npy
│       ├── ...
│       └── alms_Planck_f353.npy
├── beams
│   │── legacy
│   │   ├── leakage_beams
│   │   └── main_beams
│   └── npipe
│       ├── leakage_beams
│       └── main_beams
├── maps
│   └── planck
│       └── deep56
│           ├── legacy
│           │   ├── HFI_SkyMap_2048_R3.01_halfmission-1_f100_ivar.fits
│           │   ...
│           │   └── HFI_SkyMap_2048_R3.01_halfmission-2_f353_map_srcfree.fits
│           ├── npipe
│           │   ├── npipe6v20A_f100_ivar.fits
│           │   ...
│           │   └── npipe6v20B_f353_map_srcfree.fits
│           └── ... src subtraction stuff
├── passbands
│    └── planck
│        ├── passband_npipe_f100.dat
│        ├── passband_npipe_f143.dat
│        ├── passband_npipe_f217.dat
│        └── passband_npipe_f353.dat
│
├── best_fits
│   └── planck
│       ├── cmb_and_fg_Planck_f100xPlanck_f100.dat
│       ├── ...
│       ├── cmb.dat
│       ├── components
│       │    ├── bb_dust_Planck_f100xPlanck_f100.dat
│       │    ├── ...
│       │    └── tt_tSZxCIB_Planck_f353xPlanck_f353.dat
│       ├── fg_Planck_f100xPlanck_f100.dat
│       ├── ...
│       └── fg_Planck_f353xPlanck_f353.dat
├── catalogs
│   ├── cat_skn_090_20220526_nightonly_ordered.txt
│   └── ...
├── mcms
│    └── planck
│        ├── Planck_f100xPlanck_f100_Bbl_spin0xspin0.npy
│        └── ...
├── spectra
│   └── planck
│       ├── Dl_Planck_f100xPlanck_f100.dat
│       ├── ...
│       ├── Dl_Planck_f353xPlanck_f353.dat
├── spectra_leak_corr
│   └── planck
│       ├── Dl_Planck_f100xPlanck_f100.dat
│       ├── ...
│       ├── Dl_Planck_f353xPlanck_f353.dat
├── spectra_leak_corr_planck_bias_corr
│   └── planck
│       ├── Dl_Planck_f100xPlanck_f100.dat
│       ├── ...
│       ├── Dl_Planck_f353xPlanck_f353.dat
├── sim_spectra/
│   └── planck
│       └── Dl_Planck_f100xPlanck_f100_00_00000.dat
│       ├── ...
│       ├── Dl_Planck_f100xPlanck_f100_auto_00000.dat
│       ├── ...
│       ├── Dl_Planck_f100xPlanck_f100_cross_00000.dat
│       ├── ...
│       ├── Dl_Planck_f100xPlanck_f100_noise_00000.dat
│       └── ...
├── sim_spectra_planck_noise_and_syst/
│   └── Dl_Planck_f100xPlanck_f100_00_00000.dat
│   ├── ...
│   ├── Dl_Planck_f100xPlanck_f100_auto_00000.dat
│   ├── ...
│   ├── Dl_Planck_f100xPlanck_f100_cross_00000.dat
│   ├── ...
│   ├── Dl_Planck_f100xPlanck_f100_noise_00000.dat
│   └── ...
├── montecarlo
│   └── planck
│       └── spectra_BB_Planck_f100xPlanck_f100_auto.dat
│       ├── ...
│       └── spectra_TT_Planck_f353xPlanck_f353_noise.dat
├── planck_mc_correction
│   └── spectra_BB_Planck_f100xPlanck_f100_auto.dat
│   ├── ...
│   └── spectra_TT_Planck_f353xPlanck_f353_noise.dat
├── covariances
│   └── planck
│       └── analytic_cov_Planck_f100xPlanck_f100_Planck_f100xPlanck_f100.npy
│       ├── ...
leakage_cov_Planck_f100xPlanck_f100_Planck_f100xPlanck_f100.npy
│       ├── ...
│       └── mc_cov_Planck_f100xPlanck_f100_Planck_f100xPlanck_f100.npy
└── windows
    └── dr6xplanck
        ├── window_dr6_pa5_f150_kspace.fits
        └── ...
    └── dr6xdeep56_20251119
        ├── window_dr6_pa4_f220_baseline.fits
        ├── ...
        └── window_lat_iso_i6_f150_kspace.fits
```


# Main Pipeline
Here we get the power spectra (all possible crosses of maps) and their covariance
matrices. We may need to iterate this pipeline, e.g., initially with calibration
and polarization efficiencies of 1, in order to get the calibration and polarization
efficiencies, and then update them in the paramfile and run again (to get calibrated
spectra with properly-normalized covariances).
1. Get the apodized windows (for kspace filter and measuring spectra):
    - command: `sbatch --mem 72G --cpus-per-task 10 --time 10:00 --job-name get_windows /path/to/sbatch/script.slurm python -u /path/to/PSpipe/project/SO/pISO/python/masks/get_windows.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
2. Get best-fit signal power spectra (for spectra comparisons, e.g. calibration and 
null tests, and also for covariance matrices):
    - command: `sbatch --mem 16G --cpus-per-task 10 --time 10:00 --job-name get_best_fit_mflike /path/to/sbatch/script.slurm python -u /path/to/PSpipe/project/SO/pISO/python/get_best_fit_mflike.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
3. Get inverse mode-coupling matrices (`mbl_inv`) and so-called `Bbl` matrices. 
The `Bbl` matrices map a theoretical per-ell power spectrum to the expected
value of the data vector (i.e., the per-bin power spectrum). In the case of 
unbinned mode-coupling matrices, they are equivalent to naive binning (and so are
not really specific to any cross-spectrum, or needed at all). In the case of 
binned mode-coupling matrices, they depend on the cross-spectrum, polarization 
pair, and beam.

    The `mbl_inv` matrices are saved as `(5, nb, nl)` shaped `np.ndarray`s. They
    include the effect of the errorless main beam application/deconvolution, 
    any `Dl -> Cl -> pseudo-Dl` factors, and binning (for both the binned
    and unbinned mode-coupling matrix cases). Their ordering represents the 
    following. Given a mode-coupling matrix with block structure:
    ```
    | M_00                                   |
    |      M_02                              |
    |           M_20                         |
    |                M_++     0     0   M_-- |
    |                   0  M_++ -M_--      0 |
    |                   0 -M_--  M_++      0 |
    |                M_--     0     0   M_++ | 
    ```
    then the inverse mode-coupling matrix will have an identical block structure.
    We can denote the blocks of the inverse by `_inv`. Then, because they are
    block-diagonal, `M_00_inv = inv(M_00)`, `M_02_inv = inv(M_02)`, and 
    `M_20_inv = inv(M_20)`. The other blocks are coupled, but we use linalg
    tricks to calculate them efficiently. The `5` axis in the `mbl_inv` array
    indexes `(M_00_inv, M_02_inv, M_20_inv, M_++_inv, M_--_inv)`.

    `Bbl` is `(nb, nl)` if the mode-coupling matrix is binned after inversion,
    and follows the same `(5, nb, nl)` shape if it is binned before inversion.

    Because we need the theoretical signal pseudospectra for the covariance, to
    avoid calculating the mode-coupling matrices again in a separate script, we
    also calculate them here too.

    - command: `sbatch --ntasks 14 --cpus-per-task 8 --mem-per-cpu 8G --time 10:00 --job-name get_mcm_bbl_and_pseudosignal /path/to/sbatch/script.slurm srun python -u /path/to/PSpipe/project/SO/pISO python/get_mcm_bbl_and_pseudosignal.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
4. The power spectra often deconvolve a kspace-filter matrix which is built (for
now) at the bin-bin level. The inner product of the inverse kspace-filter matrix
and the `mbl_inv` comprise the multiplicative part of the pseudo-to-power
spectrum pipeline. This matrix also turns the pseudospectrum covariance into the 
power spectrum covariance. Therefore we construct and explicitly save these
`pseudo2datavec` matrices, which have shape `(9*nb, 9*nl)`:
    - command: `sbatch --cpus-per-task 2 --mem-per-cpu 8G --time 15:00 --job-name get_pseudo2datavec /path/to/sbatch/script.slurm srun python -u /path/to/PSpipe/project/SO/pISO/python/get_pseudo2datavec.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
5. We load and process the maps (kspace-filter them, deconvolve pixel-window 
functions, calibrate them, mask them) and calculate their binned power spectra, 
simply averaged over split crosses. This uses the `pseudo2datavec` and any
additive corrections. We also save binned auto and noise power spectra (at the
level the of per-split maps) and the per-split-cross spectra.

    The script can process either data maps or simulations. If data, we also 
    save the processed `alms` to disk, for use in measuring the empirical 
    noise pseudospectra for the covariance matrix. The mpi tasks are first 
    distributed over the maps, and then over the spectra (for `nmaps` there
    are ~ `nmaps^2/2` spectra). The `cross`, `auto`, and `noise` spectra are 
    saved as human-readable text files for each map-pair, whereas the per-split
    spectra are saved in a two-level, single-dictionary binary file with structure
    `ps_dict_all[(sv1, m1, 'sn{k}'), (sv2, m2, 'sn{k}')][pol_pair]`, where `k`
    indexes splits.

    Sims always include the CMB and Gaussian foregrounds, and `mnms` noise sims.
    Sims can include random beam or leakage systematics, and map-level lensing.
    If sims, we only save the `cross`, not the `auto` or `noise` (they are not
    needed for anything). The mpi tasks are distributed just over the sims; each
    task calculates all the alms and spectra for each sim separately. The alms
    are not saved to disk. All spectra results are saved in the binary dictionary,
    with the averaged `cross` spectra like `ps_dict_all[(sv1, m1), (sv2, m2)][pol_pair]`
    and the per-split spectra like `ps_dict_all[(sv1, m1, {spl1}), (sv2, m2, {spl2})][pol_pair]`,
    where `spl1` and `spl2` index either `s` (for the signal part of the sim)
    or `n{k}` for the per-split noise part of the sim. Keeping the signal and 
    noise separate results in a much lower-variance monte-carlo covariance
    estimate. See the `--start`, `--stop`, `--write-sim-map-start`, 
    `--write-sim-map-stop`, `--simulate-syst`, and `--simulate-lens` arguments.

    - command: `sbatch --ntasks 17 --mem 600G --cpus-per-task 4 --time 15:00 --job-name get_spectra_from_maps /path/to/sbatch/script.slurm srun python -u /path/to/PSpipe/project/SO/pISO/python/get_spectra_from_maps.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`
6. Like the pseudosignal, we need the per-ell pseudonoise spectra for the 
covariance matrix. These must be measured from the data, but fortunately, the
`alms` from the previous script trivially give us the raw noise pseudospectra.
We also smooth the estimates using a Savitzky Golay filter, specified by the 
`savgol_w` and `savgol_k` entries in the paramfile.
    - command: `sbatch --ntasks 50 --mem 500G --cpus-per-task 2 --time 15:00 --job-name get_pseudonoise /path/to/sbatch/script.slurm srun python -u /path/to/PSpipe/project/SO/pISO/python/get_pseudonoise.py /path/to/PSpipe/project/SO/pISO/paramfiles/dr6xdeep56_20251119.dict`

After this point, the `data_dir` should look like:
```bash
/path/to/my/PSpipe/data_dir
├── alms
│   └── dr6xdeep56_20251119
│       ├── alms_dr6_pa4_f220_set0.npy
│       ├── ...
│       └── alms_lat_iso_i6_f150_set3.npy
├── beams
│   ├── dr6
│   │   ├── leakage_beams
│   │   │   ├── pa4_f220_gamma_t2b.txt
│   │   │   ├── ...
│   │   │   └── pa6_f150_gamma_t2e.txt
│   │   └── main_beams
│   │       ├── coadd_pa4_f220_night_beam_tform_jitter_cmb.txt
│   │       ├── ...
│   │       └── coadd_pa6_f150_night_beam_tform_jitter_cmb.txt
│   └── lat
│       └── iso
│           ├── leakage_beams
│           └── main_beams
│               ├── beam_gaussian_fwhm_0.8_arcmin_no_error.txt
│               ├── ...
│               └── beam_gaussian_fwhm_7.4_arcmin_no_error.txt
├── best_fits
│   └── dr6xdeep56_20251119
│       ├── cmb_and_fg_dr6_pa4_f220xdr6_pa4_f220.dat
│       ├── ...
│       ├── cmb_and_fg_lat_iso_i6_f150xlat_iso_i6_f150.dat
│       ├── cmb.dat
│       ├── components
│       │   ├── bb_dust_dr6_pa4_f220xdr6_pa4_f220.dat
│       │   ├── ...
│       │   └── tt_tSZxCIB_lat_iso_i6_f150xlat_iso_i6_f150.dat
│       ├── fg_dr6_pa4_f220xdr6_pa4_f220.dat
│       ├── ...
│       ├── fg_lat_iso_i6_f150xlat_iso_i6_f150.dat
│       ├── noise
│       │   ├── pseudo_noise_dr6_pa4_f220xdr6_pa4_f220_set0.dat
│       │   ...
│       │   ├── pseudo_noise_lat_iso_i6_f150xlat_iso_i6_f150_set3.dat
│       │   ├── raw_pseudo_noise_dr6_pa4_f220xdr6_pa4_f220_set0.dat
│       │   ...
│       │   └── raw_pseudo_noise_lat_iso_i6_f150xlat_iso_i6_f150_set3.dat
│       ├── pseudo_cmb_and_fg_dr6_pa4_f220xdr6_pa4_f220.dat
│       ...
│       ├── pseudo_cmb_and_fg_lat_iso_i6_f150xlat_iso_i6_f150.dat
│       └── unlensed_cmb_and_lensing.dat
├── binning
│   └── binning_50.dat
├── maps
│   ├── dr6
│   │   ├── deep56
│   │   │   ├── act_dr6.02_std_AA_night_pa4_f220_4way_coadd_ivar.fits
│   │   │   ├── act_dr6.02_std_AA_night_pa4_f220_4way_coadd_map.fits
│   │   │   ├── act_dr6.02_std_AA_night_pa4_f220_4way_coadd_map_srcfree.fits
│   │   │   ...
│   │   │   ├── act_dr6.02_std_AA_night_pa6_f150_4way_set3_ivar.fits
│   │   │   ├── act_dr6.02_std_AA_night_pa6_f150_4way_set3_map.fits
│   │   │   └── act_dr6.02_std_AA_night_pa6_f150_4way_set3_map_srcfree.fits
│   │   └── published -> /scratch/gpfs/ACT/data/act_dr6/dr6.02/maps/published
│   └── lat
│       └── deep56
│           ├── 20251019
│           │   ├── sky_ivar_c1_f220_coadd_20251019.fits
│           │   ...
│           │   ├── sky_ivar_i6_f150_set3_20251019.fits
│           │   ├── sky_map_c1_f220_coadd_20251019.fits
│           │   ...
│           │   └── sky_map_i6_f150_set3_20251019.fits
│           └── 20251119
│               ├── sky_ivar_c1_f220_coadd_20251119.fits
│               ...
│               ├── sky_ivar_i6_f150_set3_20251119.fits
│               ├── sky_map_c1_f220_coadd_20251119.fits
│               ...
│               └── sky_map_i6_f150_set3_20251119.fits
├── mask
│   ├── dr6xdeep56_20251119
│   │   ├── xtra_mask_intersect.fits
│   │   ├── xtra_mask_lat_iso_c1_f220.fits
│   │   ...
│   │   ├── xtra_mask_lat_iso_i6_f150_set3.fits
│   │   └── xtra_mask_union.fits
│   ├── HFI_Mask_GalPlane-apo0_2048_R2.00_GAL070_fejer1.fits
│   └── source_mask_15mJy_and_dust_rad12.fits
├── mcms
│   └── dr6xdeep56_20251119
│       ├── dr6_pa4_f220xdr6_pa4_f220_Bbl.npy
│       ├── dr6_pa4_f220xdr6_pa4_f220_mode_coupling_inv.npy
│       ...
│       ├── lat_iso_i6_f150xlat_iso_i6_f150_Bbl.npy
│       ├── lat_iso_i6_f150xlat_iso_i6_f150_mode_coupling_inv.npy
│       ├── pseudo2datavec_dr6_pa4_f220xdr6_pa4_f220.npy
│       ...
│       └── pseudo2datavec_lat_iso_i6_f150xlat_iso_i6_f150.npy
├── passbands
│   ├── dr6
│   │   ├── passband_dr6_pa4_f220.dat
│   │   ├── ...
│   │   └── passband_dr6_pa6_f150.dat
│   └── lat
│       └── iso
│           ├── bandpass_mean_f090.dat
│           ├── ...
│           └── bandpass_mean_f280.dat
├── plots
│   └── dr6xdeep56_20251119
│       ├── best_fits
│       │   ├── best_fit_BB.png
│       │   ...
│       │   ├── best_fit_TT.png
│       │   ├── foregrounds_all_comps_bb.png
│       │   ...
│       │   └── foregrounds_all_comps_tt.png
│       ├── mask
│       │   ├── maps_ivar
│       │   │   ├── sky_ivar_c1_f220_set0_20251119.png
│       │   │   ...
│       │   │   ├── sky_ivar_i6_f150_set3_20251119.png
│       │   │   ├── sky_map_c1_f220_set0_20251119_0.png
│       │   │   ...
│       │   │   └── sky_map_i6_f150_set3_20251119_2.png
│       │   ├── xtra_mask_intersect.png
│       │   ├── xtra_mask_lat_iso_c1_f220.png
│       │   ...
│       │   ├── xtra_mask_lat_iso_i6_f150_set3.png
│       │   └── xtra_mask_union.png
│       ├── mcms
│       │   ├── pseudo2datavec_dr6_pa4_f220xdr6_pa4_f220.png
│       │   ├── ...
│       │   └── pseudo2datavec_lat_iso_i6_f150xlat_iso_i6_f150.png
│       ├── noise
│       │   ├── dr6_pa4_f220_Bxpa4_f220_B_set0_ps.png
│       │   ├── ...
│       │   └── lat_iso_i6_f150_Txi6_f150_T_set3_ps.png
│       └── windows
│           ├── window_dr6_pa4_f220_baseline.png
│           ...
│           ├── window_dr6_pa6_f150_kspace.png
│           ├── windowed_maps
│           │   ├── lat_iso_c1_f220_split0_Q.png
│           │   ├── lat_iso_c1_f220_split0_T.png
│           │   ├── lat_iso_c1_f220_split0_U.png
│           │   ├── ...
│           │   ├── lat_iso_i6_f150_split3_Q.png
│           │   ├── lat_iso_i6_f150_split3_T.png
│           │   └── lat_iso_i6_f150_split3_U.png
│           ├── window_lat_iso_c1_f220_baseline.png
│           ├── ...
│           └── window_lat_iso_i6_f150_kspace.png
├── spectra
│   └── dr6xdeep56_20251119
│       ├── Dl_all_sn_cross_data.npy
│       ├── Dl_dr6_pa4_f220xdr6_pa4_f220_auto.dat
│       ├── Dl_dr6_pa4_f220xdr6_pa4_f220_cross.dat
│       ├── Dl_dr6_pa4_f220xdr6_pa4_f220_noise.dat
│       ├── ...
│       ├── Dl_lat_iso_i6_f150xlat_iso_i6_f150_auto.dat
│       ├── Dl_lat_iso_i6_f150xlat_iso_i6_f150_cross.dat
│       └── Dl_lat_iso_i6_f150xlat_iso_i6_f150_noise.dat
└── windows
    └── dr6xdeep56_20251119
        ├── window_dr6_pa4_f220_baseline.fits
        ├── ...
        └── window_lat_iso_i6_f150_kspace.fits
```

Notes:
* Related to `mbl_inv`, `Bbl`, and `pseudo2datavec`, there are new functions in
`pspy` that make applying spin-weighted matrices to spectra easier:
    1. `so_mcm.get_spec2spec_array_from_spin2spin_array`, to take a `(5, ny, nx)`
    shaped spin-weighted array and build the fully-populated `(9*ny, 9*nx)`
    array for applying to per-ell spectra.
    2. `so_spectra.spec_dict2vec`, to turn a `ps[pol_pair]` dictionary into a 
    1d array.
    3. `so_spectra.vec2spec_dict`, the opposite of that.
    4. `so_spectra.spin2spin_array_matmul_spec_dict`, to apply a `(5, ny, nx)`
    shaped spin-weighted array to a `ps[pol_pair]` dictionary of spectra. Avoids
    needing to fully populate the `(9*ny, 9*nx)` array and matmul it against the
    1d array. Either option is how one would, e.g., apply `Bbl` to a vector!
* Other important new functions in `pspy` are:
    1. `so_mcm.ducc_couplings`, for fast couplings and mode-coupling matrices.
    2. `so_mcm.invert_mcm`, to use linalg tricks for fast inversion of 
    spin-weighted matrices.
