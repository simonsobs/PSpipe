**************************
Data Analysis
**************************

In this ``project`` we analyse data maps from the ACTPol experiment.


Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)


Pipeline flow chart
===================

First you need to write a dictionnary file: ``global.dict``, it will contain the parameters relevant for the Data Analysis project. Examples of ``global.dict`` are given in the ``paramfiles`` folder.
Then, we can use the codes in the ``python`` folder to run the pipeline sequentially.

First we compute the mode coupling and binning matrices associated to the ACTPol window function 

.. code:: shell

    python get_mcm_and_bbl.py global.dict

The next step is to compute all different auto and cross power spectra 

.. code:: shell

    python get_spectra.py global.dict
    
We will then compute analytical errorbars, in order to do so we need best fit signal power spectra and measured noise power power spectra

.. code:: shell

    python get_best_fit.py global.dict
    python get_noise_model.py global.dict
    
The covariance can then be obtained

.. code:: shell

    python get_covariance.py global.dict
   
For this step you might want to use mpi, at nersc, on interactive mode, it will be something like

.. code:: shell

    salloc -N 20 -C haswell -q interactive -t 03:00:00
    srun -n 20 -c 64 --cpu_bind=cores python get_covariance.py global.dict

We will put in the toeplitz approx to speed up this one very soon.
    
Finally you can compare with Choi et al spectra by running

.. code:: shell

    python compare_with_choi_spectra.py global.dict

  
