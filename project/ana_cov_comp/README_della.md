FIXME: need to merge data_analysis and ana_cov_comp, e.g., wouldn't want to have same paramfile in both projects being used in the commands
FIXME: update commands to the actual sbatch command

# Scope
These instructions are specific to generating semi-analytic covariance matrices, and do not apply to power spectra.

# Setup
1. Install `PSpipe` dependencies in an environment
2. Install `PSpipe` in a directory on your system
3. Add or modify a slurm script in the `ana_cov_comp/slurm` directory:
    - Add code that will activate your `PSpipe` environment from step 1 in the indicated block
    - Add code that will change into your `PSpipe/project/ana_cov_comp` directory from step 2 in the indicated block
    - Add any other desired slurm magic, e.g. email or redirecting slurm out files
4. Make a directory on your system for a given paramfile input and output (i.e., a given pipeline run), called the `data_dir` 
    - In a clean workspace, this should be specific to a given paramfile
5. Locate all required pre-existing products on your system. These are:
    - beams
    - maps
    - passbands
6. Either copy or symlink above products into subdirectories with those names inside the `data_dir` from step 4. Alternatively, modify your paramfile so that 
those product directories have the correct absolute path, rather than being specified relative to `data_dir`.
7. Modify your paramfile with the `data_dir` from step 4.

# Covariance Pipeline
First, we need a few products from the power spectrum pipeline. Then, we can run the scripts from the covariance pipeline. Some of these scripts only depend on the power spectrum pipeline scripts, while most of them depend on previous scripts within the covariance pipeline. For all scripts, we list their immediate dependencies, such that we can trace the full "directed acyclic graph" structure of the pipeline. Note, the pipeline is not a single branch!

Some of the below scripts have both a "recipe" mode -- this is where some metadata
about the computation in the script is calculated, but no actual computation occurs -- and a generic "compute" mode in which the actual script computation occurs. In this case, the "recipe" mode must be run first since the subsequent computations will depend on the metadata in the script "recipe."

## Products from Power Spectrum Pipeline
If you have already run these as part of the power spectrum pipeline, great! If not, or you
are only generating a covariance matrix, you will need to run them now. They may have dependencies of their own!

1. We need map-space window functions to calculate coupling matrices: 
    - command: `python data_analysis/python/get_window_dr6.py data_analysis/paramfiles/myparamfile.dict`
2. The measurements of the data alms are our only way to form the frequency-space noise model, i.e., noise power spectra:
    - *command: `python ana_cov_comp/python/get_alms.py ana_cov_comp/paramfiles/myparamfile.dict` 1h 10c 48g
    - *Note until ana_cov_comp is merged into data_analysis project, we need to run the ana_cov_comp version of `get_alms.py`. In particular, this must be run inside the `data_dir` so that it creates the `alms` directory inside the `data_dir` (for compatibility with data_analysis, the `alms` directory is defined relative to the script, rather than absolutely via the paramfile.) One might need to modify their slurm template for this job only to achieve this.
3. The next three scripts construct most of the linear operator that maps measured pseudospectra into debiased power spectra. This linear operator naturally enters the covariance calculation. First we get the mode-coupling + de-beaming + binning matrix:
    - command: `python data_analysis/python/get_mcm_and_bbl.py ana_cov_comp/paramfiles/myparamfile.dict` 2h 20c 64g
    *Same note as above regarding needing to modify the slurm template applies -- this needs to be run inside the data_dir.
4. Then we need the sims for the correction to the kspace filter:
    - command: `python data_analysis/python/kspace/mc_get_kspace_tf_spectra.py ana_cov_comp/paramfiles/myparamfile.dict`
5. Then we need the actual correction:
    - command: `python data_analysis/python/kspace/mc_kspace_tf_analysis.py ana_cov_comp/paramfiles/myparamfile.dict`

## Products for Covariance Pipeline 
The first few of these products either have no dependencies, or depend only on the above power spectrum products. Afterwards, the covariance pipeline starts to have dependencies to earlier products in the covariance pipeline. In general, one can use the `--dependency` argument of `sbatch` to submit all of the below scripts at once and let the entire pipeline progress automatically.

6. We need to get the alms of all the windows, both the baseline windows and the effective windows for the noise. Pairs of these "one point" windows are used to calculate "two point" couplings, i.e. the core piece of mode-coupling matrices:
    - depends on: 1
    - command: `python ana_cov_comp/python/get_1pt_ewin_alms.py ana_cov_comp/paramfiles/myparamfile.dict` 30m 10c 16g
7. From pairs of the "one point" window alms, calculate "two point" couplings, i.e. the core piece of mode-coupling matrices:
    - depends on: 6
    - command: `python ana_cov_comp/python/get_2pt_coupling_matrices.py ana_cov_comp/paramfiles/myparamfile.dict` 13h 20c 16g
8. We need to get the alms of all the products of two windows, both the baseline windows and the effective windows for the noise. Pairs of these "two point" windows are used to calculate "four point" couplings, i.e. the core piece of covariance matrices:
    - depends on: 1
    - command: `python ana_cov_comp/python/get_2pt_ewin_alms_and_w2.py ana_cov_comp/paramfiles/myparamfile.dict` 1h 10c 16g
9. From pairs of the "two point" window alms, calculate "four point" couplings, i.e. the core piece of covariance matrices. First we run the script in "recipe" mode to determine exactly what files need to be calculated in the first place, then we run an array of jobs in "compute" mode to actually calculate the "four point" couplings:
    - command: `python ana_cov_comp/python/get_4pt_coupling_matrices.py ana_cov_comp/paramfiles/myparamfile.dict --mode recipe` 5m 1c 4g
10. Calculate the actual "four point" couplings:
    - depends on: 8, 9
    - command: `python ana_cov_comp/python/get_4pt_coupling_matrices.py ana_cov_comp/paramfiles/myparamfile.dict` 1h 20c 16g ar 24 del 120
11. From the "two point" couplings, we do some final steps to compute mode coupling matrices and their inverses, e.g., grouping polarization coupling matrices into irreducible EB, BE blocks:
    - depends on: 7, 8
    - command: `python ana_cov_comp/python/get_mcms_and_mcm_invs.py ana_cov_comp/paramfiles/myparamfile.dict` 1h 20c 24g
12. The next three scripts deal with calculating corrections due to anisotropic structure in frequency space, e.g. the kspace filter. Anisotropies break the covariance formalism, and we fit for an ansatz correction. The first script generates a template for the correction:
    - depends on: 1
    - command: `python ana_cov_comp/python/get_effective_2pt_4pt_fl.py ana_cov_comp/paramfiles/myparamfile.dict` 30m 20c 48g
13. The second script generates a small number of mock simulations (approximate but not perfect mask, power spectra etc.) to fit the template to:
    - depends on: 12
    - command: `python ana_cov_comp/python/get_effective_2pt_4pt_spectra.py ana_cov_comp/paramfiles/myparamfile.dict` 1h 20c 24g ar 4 del 125
14. The third script fits the template to the spectra. The full "anisotropic correction" is the fitted template:
    - depends on: 13
    - command: `python ana_cov_comp/python/get_effective_2pt_4pt_fit.py ana_cov_comp/paramfiles/myparamfile.dict` 1h 10c 24g
15. We need theory signal power spectra:
    - command: `python ana_cov_comp/python/get_best_fit_mflike.py ana_cov_comp/paramfiles/myparamfile.dict` 5m 10c 4g
16. We need to transform the theory signal power spectra into normalized pseudospectra, including the anisotropic correction:
    - depends on: 11, 14, 15
    - command: `python ana_cov_comp/python/get_pseudosignal.py ana_cov_comp/paramfiles/myparamfile.dict` 30m 1c 16g
17. We need noise power spectra. These must be measured from the data:
    - depends on: 2, 11, 14
    - command: `python ana_cov_comp/python/get_noise_model.py ana_cov_comp/paramfiles/myparamfile.dict` 30m 5c 64g
18. We need to transform the noise power spectra into normalized pseudospectra, including the anisotropic correction:
    - depends on: 17
    - command: `python ana_cov_comp/python/get_pseudonoise.py ana_cov_comp/paramfiles/myparamfile.dict` 30m 1c 16g
19. Combine the signal pseudospectra, the noise pseudospectra, and the "four point" couplings into unbinned pseudocovariance matrices. These matrices are averaged over split combinations in accordance with the pseudospectra. Otherwise the exist at the lowest possible level: a combination of four fields, where each field is a survey, an array, and a polarization. To help manage computation, first we run the script in "recipe" mode, then we run an array of jobs in "compute" mode to actually calculate the pseudocovariance blocks. When calculating the recipe, we must indicate how many "groups" we want the recipe to target collecting covariance blocks into. Each "group" of blocks can run in parallel, so this is a problem of cluster resource optimization, e.g., the maximum number of cores allowed to run at once per user:
    - depends on: 9
    - command: `python ana_cov_comp/python/get_split_averaged_unbinned_pseudo_cov_blocks.py ana_cov_comp/paramfiles/myparamfile.dict --mode recipe ----target-ngroups 400` 5m 1c 4g
20. Once the recipe is complete, we can compute the actual pseudocovariance blocks. Because there are `<ngroups>` jobs, which might be large, and each job has differing resource requirements, we use a "meta" script to submit all the jobs. It is important to read the documentation of this script before submitting, especially in consideration of the cluster charges/priority, since that will affect the optimal submission arguments. It is highly recommended to run this script in `--dry-run` mode first:
    - depends on: 10, 16, 18, 19
    - command: `python ana_cov_comp/python/get_split_averaged_unbinned_pseudo_cov_blocks_submit.py ana_cov_comp/paramfiles/myparamfile.dict`
21. We need the linear operators that map pseudospectra to the (binned) power spectra:
    - depends on: 3, 4, 5
    - command: `python ana_cov_comp/python/get_pspipe_operator.py ana_cov_comp/paramfiles/myparamfile.dict` 30m 1c 24g
22. From the unbinned pseudocovariance blocks, we want to form binned power spectrum covariance blocks. This basically means sandwiching the unbinned pseudocovariance between a pair of linear operators that map the pseudospectra to the (binned) power spectra:
    - depends on: 20, 21
    - command: `python ana_cov_comp/python/get_split_averaged_binned_spec_cov_blocks.py ana_cov_comp/paramfiles/myparamfile.dict` 30m 5c 64g ar 6 delt 20
23. Now we complete the covariance matrix by stitching together all blocks in the preferred ordering:
    - depends on: 22
    - *command: `python data_analysis/python/get_xarrays_covmat.py ana_cov_comp/paramfiles/myparamfile.dict` 5m 10c 8g
    - *Note: Again, this must be run inside the `data_dir` so that it finds the `covariances` directory inside the `data_dir` (for compatibility with data_analysis, the `covariances` directory is defined relative to the script, rather than absolutely via the paramfile.) One might need to modify their slurm template for this job only to achieve this.