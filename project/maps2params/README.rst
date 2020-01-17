**************************
Maps2Parameters (internal)
**************************

In this ``project`` we generate CMB maps simulations and compute their power spectra and covariance matrices.
We then pass the data to mflike and print the chi2 with respect to the input theory curves.


Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)
* mflike : mutlifrequency likelihood interfaced with cobaya (https://github.com/simonsobs/LAT_MFLike)


Pipeline flow chart
===================

First you need to write a dictionnary file: global.dict that will contains the parameters relevant for the maps2parameter project. Example of global.dict are given in the parameter_files folder.
Then we can use the codes in the scripts folder to run the pipeline sequentially.



The first step of the pipeline is generating all the data necessary for the generation of simulation.
This includes:
- multi-frequency beam files
- multi-frequency noise power spectra
- input theory power spectra and multi-frequency fg cmb power spectra

This can be done simply by running

.. code:: shell

    python maps_to_params_prepare_sim_data.py global.dict

Then we generate the window functions and associated mode coupling matrices 

.. code:: shell

    python get_maps_to_params_mcm_and_bbl.py global.dict

The next step is to generate simulations and take their power spectra this is done with 

.. code:: shell

    python get_maps_to_params_spectra.py global.dict
    
Note that for this step and since you might want to generate many simulations, you might benefit from using a computer cluster and MPI. Details on how to run this script at NERSC with MPI are given in the NERSC section.
And analysis of the simulations can be performed by running

.. code:: shell

    python maps_to_params_mc_analysis.py global.dict

Now that we have simulations spectra, we should generate analytical covariances matrices 

.. code:: shell

    python get_maps_to_params_covariance.py global.dict

Again for this step, mpi is recommended.

We are done, with the simulations/spectra and covariances part, it's time to plot the results and check that the pipeline is working as expected

.. code:: shell

    python maps_to_params_mc_plot_spectra.py global.dict
    python maps_to_params_covariance_plot.py global.dict
