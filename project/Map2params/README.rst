Maps2Parameters (internal)
==========================

In this ``project`` we generate CMB maps simulations and compute their power spectra and covariance matrices.
We then pass the data to mflike and print the chi2 with respect to the input theory curves.


Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)
* mflike : mutlifrequency likelihood interfaced with cobaya (https://github.com/simonsobs/LAT_MFLike)


pipeline flow chart
===================

The first step of the pipeline is generating all the data necessary for the generation of simulation.
This includes:
- multi-frequency beam files
- multi-frequency noise power spectra
- input theory power spectra and multi-frequency fg cmb power spectra

This can be done simply by running

.. code:: shell

    python maps_to_params_prepare_sim_data.py global_healpix.dict

