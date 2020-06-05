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
    python get_best_fit.py global.dict
    python get_noise_model.py global.dict
    python get_covariance.py global.dict
