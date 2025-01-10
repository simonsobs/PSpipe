**************************
Planck birefringence
**************************

The goal of this pipeline is to reproduce the Minami & Komatsu `results  <https://arxiv.org/pdf/2011.11254.pdf>`_.

Some scientific doc (extracted from their papers) can be found `here <https://github.com/simonsobs/PSpipe/blob/master/project/Planck_pspy/doc/birefringence.pdf>`_.

Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)
* cobaya: python framework for sampling and statistical modelling (https://github.com/CobayaSampler/cobaya)

Pipeline flow chart
===================

First you need to write a dictionnary file: ``global.dict``, it will contain the parameters relevant for the pipeline. Examples of ``global.dict`` are given in the ``paramfiles`` folder. For birefringence, you can use in particular  ``global_EB.dict``.


Then, we can use the codes in the ``python`` and ``python/birefringence`` folder to run the pipeline sequentially.

The first step of the pipeline is to download all the public data necessary for the reproduction.
This includes:

* multi-frequency maps
* multi-frequency beam files

This can be done simply by running

.. code:: shell

    python get_planck_data.py global_EB.dict
    
Note that you will also need to copy the ``data`` folder of the ``Planck_pspy`` project at the location 
where you plan to run your scripts (it contains theory Cl and binning file).

Then we read the Minami & Komatsu window functions and compute the associated mode coupling matrices

.. code:: shell

    python get_planck_mcm_bbl.py global_EB.dict

The next step is to compute the power spectra, this is done with

.. code:: shell

    python get_planck_spectra.py global_EB.dict

We then need to model the power spectra. We need an estimate of the noise and some estimate of the underlying CMB + fg power spectra

.. code:: shell

    python EB_planck_spectra_model.py global_EB.dict

    python planck_noise_model.py global_EB.dict

Now we can generate individual analytical covariances matrices (note that for this step mpi is recommended). At nersc on interactives nodes it would be

.. code:: shell

    salloc -N 28 -C haswell -q interactive -t 01:00:00
    
    srun -n 28 -c 64 --cpu_bind=cores python EB_planck_covariance.py global_EB.dict

Then, we can rearrange the covariance elements into a multifrequency covariance matrix (keeping EB and BE separated for freq1 != freq2)

.. code:: shell

    python EB_planck_multifrequency_covmat.py global_EB.dict

Ok we are done with the data part, let's run a MCMC chain using the Minami & Komatsu likelihood

.. code:: shell

    python EB_planck_pol_angle_mf.py global_EB.dict

Finally let's plot the result

.. code:: shell

    python EB_plot_chain_results.py 
