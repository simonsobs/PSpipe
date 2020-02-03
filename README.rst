
.. raw:: html

      <img src="https://github.com/thibautlouis/PSpipe/blob/master/wiki_plot/logo.png" height="400px">

.. contents:: **Table of Contents**


The package
===========

.. image:: https://travis-ci.com/simonsobs/PSpipe.svg?branch=master
   :target: https://travis-ci.com/simonsobs/PSpipe

``PSpipe`` is a pipeline creator for the analysis of the high resolution maps of the large aperture
telescope of the Simons Observatory. It contains tools for estimating power spectra and a
multi-frequency likelihood interfaced with the ``cobaya`` MCMC sampler.

The pipelines are mainly written in python and make use of three different codes

* ``pspy`` : python library for power spectrum estimation (https://github.com/simonsobs/pspy)
* ``namaster`` : C library + python wrapper for power spectrum estimation (https://github.com/LSSTDESC/NaMaster)
* ``mflike`` : mutlifrequency likelihood interfaced with ``cobaya`` (https://github.com/simonsobs/LAT_MFLike)

The package is licensed under the BSD license.

Requirements
============

* Python >= 3.5
* `GSL <https://www.gnu.org/software/gsl/>`_: version 2 required
* `FFTW <http://www.fftw.org/>`_: version 3 required
* `cfitsio <https://heasarc.gsfc.nasa.gov/fitsio/>`_: at least version 3.0
* `healpix <https://sourceforge.net/projects/healpix/>`_: at least version 2.0

Installing
==========

Using `python/pip`
------------------

If the previous requirements are fulfilled, you can install the ``PSpipe`` package with its
dependencies by doing

.. code:: shell

   $ pip install --user git+https://simonsobs/PSpipe.git

If you plan to develop or want to use the different projects, it is better to checkout the latest
version by doing

.. code:: shell

    $ git clone https://github.com/simonsobs/PSpipe.git /where/to/clone

Then you can install the ``PSpipe`` library and its dependencies *via*

.. code:: shell

    $ pip install --user /where/to/clone
    

At ``NERSC``
------------------

    
Instructions for installing ``pspy`` at ``NERSC`` are available at  `INSTALL_PSPY_NERSC <https://github.com/simonsobs/pspy/blob/master/INSTALL_NERSC.rst>`_.

Instructions for installing ``Namaster`` at ``NERSC`` are available at  `INSTALL_NAMASTER_NERSC <https://github.com/LSSTDESC/NaMaster/blob/master/NERSC_installation.md>`_.



Using ``docker``
----------------

Given the number of requirements, you can use a ``docker`` image already made with the needed
libraries and everything compiled. You should first install `docker
<https://docs.docker.com/install/>`_ for your operating system.  


Bash installation
~~~~~~~~~~~~~~~~~~

We have written a simple bash script to install the ``PSpipe`` docker and to clone the main ``PSpipe`` libraries.
Just run 

.. code:: shell

   $ sh start_docker.sh
  
after specifying ``/WHERE/TO/WORK/pspipe_workspace``, which is the folder in which you want to be able to work with ``PSpipe``. 



This will open a new ``bash`` terminal with a full installation of ``PSpipe``, ``pixell``,
``NaMaster``, ``pspy``... For instance, you can start the ``ipython`` interpreter and run the following
``import`` command

.. code:: shell

   $ ipython
   Python 3.6.9 (default, Nov  7 2019, 10:44:02)
   Type 'copyright', 'credits' or 'license' for more information
   IPython 7.11.1 -- An enhanced Interactive Python. Type '?' for help.

   In [1]: import pixell, pymaster, pspy

You can run the python scripts from the tutorials directory of ``PSpipe``.

When you are done with the image, just type ``exit`` and you will go back to your local machine prompt.

Running ``jupyter`` notebook from docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to start a ``jupyter`` server from the ``PSpipe`` image and run it into your web
browser.  Inside the image terminal, you have to start the ``jupyter`` server by typing

.. code:: shell

   $ jupyter notebook --ip 0.0.0.0 ~/PSpipe/notebooks

Finally open the ``http`` link (something like ``http://127.0.0.1:8888/?token...``) within your web
browser and you should be able to run one of the ``python`` notebook.

Sharing data between the ``docker`` container and the host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything perfomed within the ``/home/pspipe/workspace`` directory will be reflected into
the ``/WHERE/TO/WORK/pspipe_workspace`` on your host machine. You can then share configuration files, source codes, data
files... between the running ``docker`` container and your local machine. Nothing will be lost after
you exit from the ``docker`` container.
