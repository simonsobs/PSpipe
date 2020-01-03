
.. raw:: html

      <img src="https://github.com/thibautlouis/PSpipe/blob/master/wiki_plot/logo.png" height="400px">

.. contents:: **Table of Contents**

The package
===============
PSpipe is a package to extract cosmological parameters information from the high resolution maps of the large aperture telescope of the Simons Observatory. It contains tools for estimating power spectra and a multi-frequency likelihood interfaced with the cobaya MCMC sampler.


Requirements
===============
The pipeline is mainly written in python and make use of three different codes

* pspy : python library for power spectrum estimation (https://github.com/thibautlouis/pspy)
* namaster : C library + python wrapper for power spectrum estimation (https://github.com/LSSTDESC/NaMaster)
* mflike : mutlifrequency likelihood interfaced with cobaya (https://github.com/simonsobs/LAT_MFLike)


