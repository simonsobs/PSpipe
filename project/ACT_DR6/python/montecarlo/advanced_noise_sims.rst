********************************************
Running Tiled and Wavelet Noise Simulations
********************************************

In this note, we will describe how to run the tiled and wavelet noise sims. These instructions 
will be focused on NERSC, but should be similar for other clusters like Tiger/Della/Niagara.
The parameter file we have is focused on ACT DR6, ``paramfiles/global_dr6_v3_4pass_pa6_gaussian.dict``, 
and you should follow the standard pipeline instructions to generate windows, mode-coupling matrices, 
spectra, and a noise model from the real maps.

Note that the tiled and wavelet sims can only be run on a single array at a time, at present. 
Correlations on the same wafer are assumed to dominate over correlations between different wafers.

At present, you need to install a few different codes to run these.

1. Pull/reinstall the master branch of https://github.com/ACTCollaboration/mnms
2. Pull/reinstall the prelim_dr6 branch of https://github.com/simonsobs/soapack/tree/prelim_dr6/soapack. Important that you are on the prelim_dr6 branch!
3. Pull/reinstall the newest optweight package https://github.com/AdriJD/optweight/tree/master/optweight
4. Note that optweight requires https://github.com/amaurea/enlib/ (you can use the cca_intel config on NERSC)

Next, you will need to create a file called ``.soapack.yml`` in your home directory. Here is a minimal setup on NERSC

.. code-block::

    dr6:
        coadd_input_path: "//global/homes/s/sigurdkn/project/actpol/maps/dr6v2_20200624/release/"
        coadd_beam_path: "/global/homes/a/adriaand/project/actpol/20210629_beams/release/"
        mask_path: "/project/projectdirs/act/data/masks/"
        default_mask_version: "masks_20200723"
        day_coadd_beam_path: "/project/projectdirs/act/data/synced_beams/ibeams_2019/"
        planck_path: "/global/cfs/cdirs/act/data/synced_maps/planck_hybrid/"
        wmap_path: ""
        default_mask_version: "masks_20200723"

    dr6v3:
        coadd_input_path: "/global/project/projectdirs/act/data/sigurdkn/actpol/maps/dr6v3_20211031/release/"
        coadd_output_path: "/global/cfs/cdirs/act/data/synced_maps/imaps_2019/"
        coadd_beam_path: "/global/cfs/cdirs/act/data/synced_beams/ibeams_2019/"
        planck_path: "/global/cfs/cdirs/act/data/synced_maps/planck_hybrid/"
        wmap_path: ""
        mask_path: "/project/projectdirs/act/data/masks/"
        default_mask_version: "masks_20200723"

    mnms:
        maps_path: "/global/cscratch1/sd/xzackli/sims/"
        covmat_path: "/global/cfs/cdirs/act/data/zatkins/mnms/release/20211203/covmats/"
        mask_path: "/global/cfs/cdirs/act/data/zatkins/mnms/release/20211203/masks/"
        default_data_model: "dr6v3"
        default_mask_version: "v3"

If you want to save some sim alms eventually, you can change ``maps_path`` under the ``mnms`` category. 

Once you have this file, request an interactive node

.. code:: shell

    salloc -N 1 -C haswell -q interactive -t 01:00:00

Once you have an interactive node ready, set up the environment. 
**You can't just paste the following! You need to change some things for your own environment.**

.. code:: shell

    module load python intel
    source activate YOUR_CONDA_ENV
    export PYTHONPATH=YOUR_PSPIPE/data_analysis/python/:YOUR_ENLIB_DIRECTORY:$PYTHONPATH
    export HDF5_USE_FILE_LOCKING=FALSE
    export OMP_NUM_THREADS=32

Activate your conda environment as necessary. Add the PSpipe scripts to your 
``PYTHONPATH`` as usual, but also add the enlib directory! The HDF5 flag is required for 
the wavelet sims due to a quirk of NERSC. The ``OMP_NUM_THREADS`` will provide threading support, 
and you should change it if you want to use a different node request.

Now finally you can run the noise sim script,

.. code:: shell

    srun -n 1 -c 32 --cpu_bind=cores python -u python/montecarlo/mc_mnms_get_spectra.py paramfiles/global_dr6_v3_4pass_pa6_gaussian.dict

The important parameters are ``noise_sim_type`` which can be ``gaussian``, ``tiled``, or ``wavelet``, and the 
dictionary for ``noise_model_parameters``. You can override the noise_sim_type with the command line option ``--noisetype tiled``, or ``--noisetype wavelet``.

We also provide an example slurm script, ``submit_noise_sim.slurm``, which you can run with 

.. code:: shell

    sbatch submit_noise_sim.slurm gaussian
    sbatch submit_noise_sim.slurm tiled
    sbatch submit_noise_sim.slurm wavelet

This particular script runs 128 realizations of the chosen sim provided in the SLURM script, by 
performing 8 sims at a time. 
