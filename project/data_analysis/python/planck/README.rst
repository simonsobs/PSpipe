*******************
Planck maps pre-processing
*******************

This directory holds tools to pre-process Planck NPIPE maps used to perform calibration and consistency tests.

The first step is the projection Planck NPIPE maps publicly available at ``NERSC`` into a CAR pixellization. To project 100,143,217 and 353 GHz maps (8 maps), one should run

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 02:00:00
    srun -n 8 -c 32 --cpu_bind=cores python project_planck_npipe_maps.py planck_pre_processing.dict

Once projected we might want to use them to compute power spectra or to get source subtracted maps. It requires to get data products such as beams which can be written to disk with

.. code:: shell

    python get_planck_npipe_beams.py

To perform source subtraction using `dory`, one needs to get a single frequency source catalog by running ``reformat_source_catalog.py`` that produces three catalogs from the ACT DR6 multi-frequency catalog

.. code:: shell

    python reformat_source_catalog.py

The source subtraction can then be performed at ``NERSC`` using the following instructions

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 02:30:00
    ./run_npipe_src_subtraction.sh

and the source subtraction process can be checked by running the ``check_src_subtraction.py`` script which displays the residual maps around point sources.

This directory also provides tools to get noise alms from NPIPE simulations to get montecarlo corrected errors. This script writes the alms in the same format as the ``mnms`` noise alms such that we can directly use the standard simulation script to get Planck simulations.

.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 01:00:00
    srun -n 64 -c 4 --cpu_bind=cores python get_planck_npipe_sim_nlms.py
    #real 26m42.475s (128 sims at 100,143,217 GHz)
