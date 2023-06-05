**************************
Computing DR6 spectra
**************************


Requirements
============

* pspy >= 1.6.1
* pspipe_utils >= 0.1.3

dependency: pspy, pspipe_utils

Here are some specific instructions to compute spectra for DR6 at NERSC.
Since it is a lot of spectra computation, we are going to make full use of MPI capacities.
The current dictionnary is called ``global_dr6.dict`` and is given in the ``paramfiles`` folder.
Then, we can use the codes in the ``python`` folder to run the pipeline sequentially.
Here we give instructions to install and to run the full thing on interactive nodes, you can of
course also submit it to NERSC standard nodes

Installation steps
------------------

First, we strongly recommand to install everything in a virtual ``python`` environment in order to
avoid clash with other ``python`` modules installed, for instance, within the ``.local``
directory. You can use the following script to install everything (the ``mpi4py`` installation
command is especially important @ NERSC)

.. code:: shell

    base_dir=~/pspipe

    slurm_account=mp107
    export SBATCH_ACCOUNT=${slurm_account}
    export SALLOC_ACCOUNT=${slurm_account}

    module load python


    pyenv_dir=${base_dir}/pyenv/perlmutter
    if  [ ! -d ${pyenv_dir} ]; then
        python -m venv ${pyenv_dir}
        source ${pyenv_dir}/bin/activate
        python -m pip install -U pip wheel
        python -m pip install ipython
        python -m pip install numpy
        module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
        MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
    fi

    software_dir=${base_dir}/software
    if  [ ! -d ${software_dir} ]; then
        mkdir -p ${software_dir}
        (
            cd ${software_dir}
            git clone git@github.com:simonsobs/PSpipe.git
            cd PSpipe
            python -m pip install -e .
        )
    fi

    source ${pyenv_dir}/bin/activate

The ``base_dir`` is where everything (virtual env. and ``pspipe`` scripts) will be located. Save the
above commands within a ``setup.sh`` file and run it with

.. code:: shell
    source setup.sh

The first time you run the script, it will install everything. Every time you log to NERSC machines,
you **need to source this file** with ``source setup.sh`` to get into the virtual environment and
use the proper software suite.

Running the pipeline
--------------------

First we need to create all the window functions. In the following we will assume that the window functions  used in temperature and in polarisation are the same, we will create the windows based on a the edges of the survey, a galactic mask, a pt source mask and a threshold on the amount of crosslinking in the patch of observation. For n seasons with m dichroic arrays (mx2 frequency maps), we will have N = n x m x 2  window functions.

.. code:: shell

    salloc -N 6 -C haswell -q interactive -t 01:00:00
    srun -n 6 -c 64 --cpu_bind=cores python get_window_dr6.py global_dr6_v3_4pass.dict

The next step is to precompute the mode coupling matrices associated with these window functions, we have N window functions corresponding to each (season X array a) data set, we will have to compute all the cross power spectra of the form
(season X array 1)  x (season Y array 2) there are therefore Ns = N * (N+1)/2 independent spectra to compute

.. code:: shell

    salloc -N 21 -C haswell -q interactive -t 00:30:00
    srun -n 21 -c 64 --cpu_bind=cores python get_mcm_and_bbl.py global_dr6_v3_4pass.dict

Now we can compute all the power spectra, the mpi loop is done on all the different arrays.
If you consider six detector arrays, we first compute the alms using mpi, and then have a simple code to combine them into power spectra

.. code:: shell

    salloc -N 6 -C haswell -q interactive -t 04:00:00
    srun -n 6 -c 64 --cpu_bind=cores python get_alms.py global_dr6_v3_4pass.dict
    srun -n 6 -c 64 --cpu_bind=cores python get_spectra_from_alms.py global_dr6_v3_4pass.dict


Finally, we need to compute the associated covariances of all these spectra, for this we need a model for the signal and noise power spectra

.. code:: shell

    salloc -N 1 -C haswell -q interactive -t 00:30:00
    srun -n 1 -c 64 --cpu_bind=cores python get_best_fit_mflike.py global_dr6_v3_4pass.dict
    srun -n 1 -c 64 --cpu_bind=cores python get_noise_model.py global_dr6_v3_4pass.dict

The computation of the covariance matrices is then divided into two steps, first compute all (window1 x window2) alms needed for the covariance computation, then the actual computation, note that there is Ns(Ns+1)/2 covariance matrix block to compute, this is enormous and is therefore the bottleneck of the spectra computation.


.. code:: shell

    salloc -N 40 -C haswell -q interactive -t 04:00:00
    srun -n 40 -c 64 --cpu_bind=cores python get_sq_windows_alms.py global_dr6_v3_4pass.dict
    srun -n 40 -c 64 --cpu_bind=cores python get_covariance_blocks.py global_dr6_v3_4pass.dict

Uncertainties in the beam of the telescope need to be propagated, the covariance matrix associated to beam errors can be computed analytically as

.. code:: shell

    salloc -N 6 -C haswell -q interactive -t 04:00:00
    srun -n 6 -c 64 --cpu_bind=cores python get_beam_covariance.py global_dr6_v3_4pass.dict

To get accurate transfer function estimated from simulation

.. code:: shell

    salloc -N 40 -C haswell -q interactive -t 04:00:00
    srun -n 40 -c 64 --cpu_bind=cores python mc_get_kspace_tf_spectra.py global_dr6_v3_4pass.dict
    salloc -N 1 -C haswell -q interactive -t 01:00:00
    srun -n 1 -c 64 --cpu_bind=cores python mc_tf_analysis.py global_dr6_v3_4pass.dict


We have also implemented a simple simulation pipeline to check if the pipeline produce unbiased spectra and accurate analytical covariance matrices
to run it

.. code:: shell

    salloc -N 40 -C haswell -q interactive -t 04:00:00
    srun -n 40 -c 64 --cpu_bind=cores python mc_get_spectra.py global_dr6_v3_4pass.dict

if you wants to rather use mnms sims:

.. code:: shell

    salloc -N 40 -C haswell -q interactive -t 04:00:00
    srun -n 40 -c 64 --cpu_bind=cores python mc_mnms_get_spectra.py global_dr6_v3_4pass.dict


then to analyze and plot the simulations

.. code:: shell

    salloc -N 1 -C haswell -q interactive -t 01:00:00
    srun -n 1 -c 64 --cpu_bind=cores python mc_analysis.py global_dr6_v3_4pass.dict
    srun -n 1 -c 64 --cpu_bind=cores python mc_cov_analysis.py global_dr6_v3_4pass.dict
    srun -n 1 -c 64 --cpu_bind=cores python mc_plot_spectra.py global_dr6_v3_4pass.dict
    srun -n 1 -c 64 --cpu_bind=cores python mc_plot_covariances.py global_dr6_v3_4pass.dict





We can now combine the data together, for this we run

.. code:: shell

    salloc -N 1 -C haswell -q interactive -t 04:00:00
    srun -n 1 -c 64 --cpu_bind=cores python get_xarrays_covmat.py global_dr6_v3_4pass.dict
    srun -n 1 -c 64 --cpu_bind=cores python get_xfreq_spectra.py global_dr6_v3_4pass.dict


We are done !
