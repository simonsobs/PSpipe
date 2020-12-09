**************************
Planck pspy
**************************

In this ``project`` we compute the power spectra and covariances matrices of the Planck data.

As a bonus, you can have a look at `README_biref  <https://github.com/simonsobs/PSpipe/blob/master/project/Planck_pspy/README_biref.rst>`_ for a preliminary reproduction of the Minami & Komatsu `results  <https://arxiv.org/pdf/2011.11254.pdf>`_.

Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)

Pipeline flow chart
===================

First you need to write a dictionnary file: ``global.dict``, it will contain the parameters relevant for the Planck project. Examples of ``global.dict`` are given in the ``paramfiles`` folder.
Then, we can use the codes in the ``python`` folder to run the pipeline sequentially.

The first step of the pipeline is to download all the public data necessary for the reproduction.
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

    python planck_best_fit.py global.dict

    python planck_noise_model.py global.dict

Now that we have an estimate of the noise, we should generate analytical covariances matrices 

.. code:: shell

    python get_planck_covariance.py global.dict


Simulation pipeline
===================

We also have a pipeline to generate Planck simulations, either based on the measured noise properties of planck data or on ffp10 simulations (the choice happens in the global.dict). 

First we use planck public chains result to generate the best fit power spectra for the different frequencies of observation of Planck 


.. code:: shell

    python planck_best_fit.py global.dict

From best fits, the planck noise model and beam model we generate gaussian simulations of planck data (note that we also have an option for using the ffp10 noise sims)

.. code:: shell

    python planck_sim_spectra.py global.dict
    
Then we analyse the simulations, check that their mean agree with  input power spectra and that their std agree with planck analytical covariance matrix 

.. code:: shell

    python planck_sim_analysis.py global.dict

Finally we make a bunch of null tests, comparing TE and ET using monte carlo errorbars

.. code:: shell

    python planck_sim_null_test.py global.dict


