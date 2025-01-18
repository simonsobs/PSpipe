
.. raw:: html

      <img src="https://github.com/simonsobs/PSpipe/blob/master/docs/logo.png" height="400px">


The package
===========

.. image:: https://img.shields.io/github/actions/workflow/status/simonsobs/pspipe/testing.yml?branch=master
   :target: https://github.com/simonsobs/pspipe/actions?query=workflow%3ATesting
.. image:: https://img.shields.io/badge/license-BSD-yellow
   :target: https://github.com/simonsobs/pspipe/blob/master/LICENSE

``PSpipe`` is a pipeline creator for the analysis of the high resolution maps of the large aperture
telescope of the Simons Observatory. It contains tools for estimating power spectra and covariance
matrices.

The pipelines are mainly written in python and make use of three different codes,

* ``pspy`` : a python library for power spectrum estimation (https://github.com/simonsobs/pspy)
* ``pspipe_utils`` : a python toolbox library to process and to deal with power spectrum computation
  (https://github.com/simonsobs/pspipe_utils)
* ``mflike`` : a mutlifrequency likelihood interfaced with ``cobaya``
  (https://github.com/simonsobs/LAT_MFLike)

The package is licensed under the BSD license.

Installing
==========

Using `python/pip`
------------------

You can install the ``PSpipe`` package with its dependencies by doing

.. code:: shell

   pip install --user git+https://github.com/simonsobs/PSpipe.git

If you plan to develop or want to use the different projects, it is better to checkout the latest
version by doing

.. code:: shell

   git clone https://github.com/simonsobs/PSpipe.git /where/to/clone

Then you can install the ``PSpipe`` library and its dependencies *via*

.. code:: shell

   pip install --user /where/to/clone
