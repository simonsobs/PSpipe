**************************
Planck
**************************

In this ``project`` we compute the power spectra and covariances matrices of the Planck data.

Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)

Pipeline flow chart
===================

First you need to write a dictionnary file: ``global.dict``, it will contain the parameters relevant for the Planck project. Examples of ``global.dict`` are given in the ``paramfiles`` folder.
Then, we can use the codes in the ``python`` folder to run the pipeline sequentially.

The first step of the pipeline is to download all the public data planck necessary for the reproduction.
This includes:

* multi-frequency maps
* multi-frequency beam files
* multi-frequency likelihood masks

This can be done simply by running

.. code:: shell

    python get_planck_data.py global.dict

then we generate the window functions and associated mode coupling matrices 

.. code:: shell

    python get_planck_mcm_bbl.py global.dict

The next step is to compute the power spectra this is done with

.. code:: shell

    python get_planck_spectra.py global.dict

we then need to model the power spectra in particular we need an estimate of the noise

.. code:: shell

    python model_planck_spectra.py global.dict

Now that we have simulations spectra, we should generate analytical covariances matrices 

.. code:: shell

    python get_planck_covariance.py global.dict

