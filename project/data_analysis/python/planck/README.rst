*******************
Planck maps pre-processing
*******************

This directory holds tools to pre-process Planck NPIPE maps and Planck legacy maps used to perform calibration and consistency tests.

The first step is the projection Planck NPIPE/legacy maps publicly available at ``NERSC`` into a CAR pixellization. To project 100,143,217 and 353 GHz maps (8 maps), one should run

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 02:00:00
    srun -n 8 -c 32 --cpu_bind=cores python project_planck_maps.py

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

This directory also provides tools to get noise alms from NPIPE/legacy simulations to get montecarlo corrected errors. This script writes the alms in the same format as the ``mnms`` noise alms such that we can directly use the standard simulation script to get Planck simulations.

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 01:00:00
    srun -n 64 -c 4 --cpu_bind=cores python get_planck_npipe_sim_nlms.py global_dr6v4xnpipe.dict
    #real 26m42.475s (128 sims at 100,143,217 GHz)

Note that the Planck Npipe spectra are biased, and we do need to correct the bias before comparing them to
the ACT spectra, the way we get the correction is the following

.. code:: shell

    salloc -N 4 -C cpu -q interactive -t 03:00:00
    srun -n 32 -c 32 --cpu_bind=cores python get_planck_spectra_correction_from_nlms.py global_dr6v4xnpipe.dict

    salloc -N 1 -C cpu -q interactive -t 01:00:00
    srun -n 1 -c 256 --cpu_bind=cores python mc_analysis.py global_dr6v4xnpipe.dict

The first code is similar to the standard simulation spectra code, but it's residual only (no signal), the mc_analysis serve to produce the average of these spectra, then to correct the planck spectra run

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 01:00:00
    srun -n 1 -c 256 --cpu_bind=cores python get_corrected_planck_spectra.py global_dr6v4xnpipe.dict
