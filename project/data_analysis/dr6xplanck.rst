*******************
Planck maps pre-processing
*******************

Here are some specific instructions to pre-process Planck maps, namely to project them and to subtract the planck bright sources

The first step is the projection Planck NPIPE/legacy maps publicly available at ``NERSC`` into a CAR pixellization. To project 100,143,217 and 353 GHz maps (8 maps), one should run

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 02:00:00
    OMP_NUM_THREADS=32 srun -n 8 -c 32 --cpu_bind=cores python project_planck_maps.py

Once projected we might want to use them to compute power spectra or to get source subtracted maps. It requires to get data products such as beams which can be written to disk with

.. code:: shell

    python extract_planck_beam.py

To perform source subtraction using `dory`, one needs to get a single frequency source catalog by running ``reformat_source_catalog.py`` that produces three catalogs from the ACT DR6 multi-frequency catalog

.. code:: shell

    python reformat_source_catalog.py

The source subtraction can then be performed at ``NERSC`` using the following instructions for npipe or legacy maps. You will have to update the `dory_path` field to point to your local `tenki` library.

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 02:30:00
    ./run_npipe_src_subtraction.sh
    ./run_legacy_src_subtraction.sh

and the source subtraction process can be checked by running the ``check_src_subtraction.py`` script which displays the residual maps around point sources.

*******************
ACT/Planck cross correlation
*******************

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_window_dr6.py global_dr6v4xlegacy.dict
    # real    10m2.348s

    OMP_NUM_THREADS=20 srun -n 12 -c 20 --cpu-bind=cores python get_mcm_and_bbl.py global_dr6v4xlegacy.dict
    # real    3m34.684s

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=32 srun -n 8 -c 32 --cpu-bind=cores python get_alms.py global_dr6v4xlegacy.dict
    # real    6m11.321s
    OMP_NUM_THREADS=32 srun -n 8 -c 32 --cpu-bind=cores python get_spectra_from_alms.py global_dr6v4xlegacy.dict
    # real    12m6.917s

.. code:: shell

    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_best_fit_mflike.py global_dr6v4xlegacy.dict
    # real    1m56.482s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_noise_model.py global_dr6v4xlegacy.dict
    # real    4m4.662s

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=36 srun -n 7 -c 36 --cpu-bind=cores python get_sq_windows_alms.py global_dr6v4xlegacy.dict
    # real 1m15.901s
    salloc --nodes 4 --qos interactive --time 03:00:00 --constraint cpu
    OMP_NUM_THREADS=32 srun -n 32 -c 32 --cpu-bind=cores python get_covariance_blocks.py global_dr6v4xlegacy.dict
    # real    13m24.803s
    
to correct for the leakage, grab the code in the leakage folder

.. code:: shell

    salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_leakage_corrected_spectra_per_split.py global_dr6v4xlegacy.dict
    # real 4m9.442s
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_leakage_sim.py global_dr6v4xlegacy.dict
    # real 20m20.127s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_leakage_covariance.py global_dr6v4xlegacy.dict
    # real 18m12.066s


the planck spectra can have leftover systematic in them, we have estimated this using planck end-to-end simulations (see bottom of this page), grab the code in the planck folder

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python get_corrected_planck_spectra.py global_dr6v4xlegacy.dict

Now to calibrate and get the expected polarisation efficiencies, grab the code in the calibration folder

.. code:: shell

    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python get_calibs.py global_dr6v4xlegacy.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python get_polar_eff_LCDM.py global_dr6v4xlegacy.dict

In addition to the standard dr6 simulation tools (e.g:)

.. code:: shell

    salloc --nodes 2 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=128 srun -n 4 -c 128 --cpu_bind=cores python mc_mnms_get_nlms.py global_dr6_v4.dict
    # real time ~ 3h (for 100 sims)

we have code to get planks simulation nlms

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 03:00:00
    OMP_NUM_THREADS=4 srun -n 64 -c 4 --cpu_bind=cores python get_planck_sim_nlms.py global_dr6v4xlegacy.dict
    #real 26m42.475s (300 sims at 100, 143, 217 GHz)

you can then use the usual monte-carlo code to generate simulated spectra

.. code:: shell

    salloc --nodes 4 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu_bind=cores python mc_mnms_get_spectra_from_nlms.py global_dr6v4xlegacy.dict
    
You can analyse and plot the simulation results using

.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_analysis.py global_dr6_v4.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_cov_analysis.py global_dr6_v4.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_plot_spectra.py global_dr6_v4.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_plot_covariances.py global_dr6_v4.dict




*******************
End-to-end sim correction
*******************


Note that the Planck Npipe spectra are biased, and we do need to correct the bias before comparing them to
the ACT spectra, the way we get the correction is the following

.. code:: shell

    salloc -N 4 -C cpu -q interactive -t 03:00:00
    OMP_NUM_THREADS=32 srun -n 32 -c 32 --cpu_bind=cores python get_planck_spectra_correction_from_nlms.py global_dr6v4xnpipe.dict

    salloc -N 1 -C cpu -q interactive -t 01:00:00
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_analysis.py global_dr6v4xnpipe.dict

The first code is similar to the standard simulation spectra code, but it's residual only (no signal), the mc_analysis serve to produce the average of these spectra.

*******************
Comparison of ACT and Planck
*******************

In order to compare ACT and Planck power spectrum, once you have computed both ACTxNpipe and ACTxlegacy, grab the script in the planck folder and run

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 00:30:00
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python AxP_comparison.py global_dr6v4xlegacy.dict

note that you have to specify in this script the location of the spectra and covariances of the npipe and legacy run.
