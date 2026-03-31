**************************
Redoing SPT analysis
**************************

In this pipeline, I am putting a bunch of scripts that can be used to redo the SPT analysis at NERSC.
This is currently in dev, note that you need pspy version 1.8.4 to run it.


Running the main pipeline
-------------------------------------------------------

.. code:: shell

    salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu
    OMP_NUM_THREADS=32 srun -n 6 -c 32 --cpu-bind=cores python spt_get_mcm_and_bbl.py global_spt.dict
    OMP_NUM_THREADS=64 srun -n 3 -c 64 --cpu-bind=cores python spt_get_alms.py global_spt.dict
    OMP_NUM_THREADS=64 srun -n 3 -c 64 --cpu-bind=cores python spt_get_spectra_from_alms.py global_spt.dict


The pipeline take roughly 30 minutes, all time is spent doing map2alms since spt has 30 splits per frequency.
An alternative dictfile using half-missions is provided that allows for much faster investigations.

You can then compare with the spt released spectra by running spt_plot_spectra.py, note that you need to install candl and spt_candl_data first.

.. code:: shell

    pip install candl-like
    git clone https://github.com/SouthPoleTelescope/spt_candl_data.git
    cd spt_candl_data
    pip install .


