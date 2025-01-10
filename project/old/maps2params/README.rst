**************************
Maps2Parameters
**************************

In this ``project`` we generate CMB simulations and compute their power spectra and covariance matrices.
We then pass the data to mflike and compute the chi2 with respect to the input theory curves.


Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)
* mflike : mutlifrequency likelihood interfaced with cobaya (https://github.com/simonsobs/LAT_MFLike)


Pipeline flow chart
===================

First you need to write a dictionnary file: ``global.dict``, it will contain the parameters relevant for the maps2parameter project. Examples of ``global.dict`` are given in the ``paramfiles`` folder.
Then, we can use the codes in the ``python`` folder to run the pipeline sequentially.

The first step of the pipeline is generating all the data necessary for the generation of simulations.
This includes:

* multi-frequency beam files
* multi-frequency noise power spectra
* input theory power spectra and multi-frequency fg cmb power spectra

This can be done simply by running

.. code:: shell

    python maps_to_params_prepare_sim_data.py global.dict

then we generate the window functions and associated mode coupling matrices

.. code:: shell

    python get_maps_to_params_mcm_and_bbl.py global.dict

The next step is to generate simulations and take their power spectra. For this step you might want to generate many simulations and you might benefit from using a computer cluster with MPI. Details on how to run this script at NERSC with MPI are given in the NERSC section.
This is done with

.. code:: shell

    salloc -N 40 -C haswell -q interactive -t 01:00:00
    srun -n 40 -c 64 --cpu_bind=cores python get_maps_to_params_spectra.py global.dict

And analysis of the simulations can be performed by running

.. code:: shell

    python maps_to_params_mc_analysis.py global.dict

Now that we have simulations spectra, we should generate analytical covariances matrices. Again for this step, mpi is recommended.

.. code:: shell

    salloc -N 21 -C haswell -q interactive -t 02:00:00
    srun -n 21 -c 64 --cpu_bind=cores python get_maps_to_params_covariances.py global.dict

We are done with the simulations/spectra and covariances part, it is time to plot the results and check that the pipeline is working as expected

.. code:: shell

    python maps_to_params_mc_plot_spectra.py global.dict
    python maps_to_params_mc_plot_covariances.py global.dict

Now you might want to produce the ``sacc`` files for your set of simulations. This can be done with

.. code:: shell

    python maps_to_params_multifrequency_covmat.py global.dict
    python maps_to_params_port_to_sacc.py global.dict
