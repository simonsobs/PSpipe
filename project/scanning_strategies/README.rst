**************************
Scanning strategy
**************************

In this ``project`` we compute the noise and covariance properties corresponding to the different SO scanning strategy


Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)


Pipeline flow chart
===================


Let's assume you run the script on NERSC, first create a folder and copy the code, then compute the spectra using


.. code:: shell

    salloc -N 5 -C haswell -q interactive -t 01:00:00
    srun -n 5 -c 64 --cpu_bind=cores python SO_noise_get_spectra_CAR.py global_CAR.dict
    
you can plot the spectra and get the covariance with

.. code:: shell

    salloc -N 1 -C haswell -q interactive -t 04:00:00
    srun -n 1 -c 64 --cpu_bind=cores python SO_noise_plot_spectra.py global_CAR.dict
    srun -n 1 -c 64 --cpu_bind=cores python SO_noise_get_covariance.py global_CAR.dict
    srun -n 1 -c 64 --cpu_bind=cores python SO_noise_plot_covariance.py global_CAR.dict
