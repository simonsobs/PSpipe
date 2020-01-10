
.. raw:: html

      <img src="https://github.com/thibautlouis/PSpipe/blob/master/wiki_plot/logo.png" height="400px">

.. contents:: **Table of Contents**

The package
-----------

``PSpipe`` is a pipeline creator for the analysis of the high resolution maps of the large aperture
telescope of the Simons Observatory. It contains tools for estimating power spectra and a
multi-frequency likelihood interfaced with the ``cobaya`` MCMC sampler.

The pipelines are mainly written in python and make use of three different codes

* ``pspy`` : python library for power spectrum estimation (https://github.com/simonsobs/pspy)
* ``namaster`` : C library + python wrapper for power spectrum estimation (https://github.com/LSSTDESC/NaMaster)
* ``mflike`` : mutlifrequency likelihood interfaced with ``cobaya`` (https://github.com/simonsobs/LAT_MFLike)

Requirements
------------

* Python >= 3.5
* `GSL <https://www.gnu.org/software/gsl/>`_: version 2 required
* `FFTW <http://www.fftw.org/>`_: version 3 required
* `cfitsio <https://heasarc.gsfc.nasa.gov/fitsio/>`_: at least version 3.0
* `HEALPix <https://sourceforge.net/projects/healpix/>`_: at least version 2.0
