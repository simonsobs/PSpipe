**************************
Computing DR6 spectra
**************************


Requirements
============

* pspy >= 1.6.1
* pspipe_utils >= 0.1.4

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

    slurm_account=mp107
    export SBATCH_ACCOUNT=${slurm_account}
    export SALLOC_ACCOUNT=${slurm_account}

    module load python
    module load intel

    base_dir=/path/to/base/dir

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
            git clone git@github.com:simonsobs/pspy.git
            cd pspy
            python -m pip install -e .
        )
        (
            cd ${software_dir}
            git clone git@github.com:simonsobs/pspipe_utils.git
            cd pspipe_utils
            python -m pip install -e .
        )
        (
            cd ${software_dir}
            git clone git@github.com:simonsobs/PSpipe.git
            cd PSpipe
            python -m pip install -e .
        )
        (
            cd ${software_dir}
            git clone git@github.com:AdriJD/optweight.git
            cd optweight
            python -m pip install -e .
        )
        (
            cd ${software_dir}
            git clone git@github.com:amaurea/enlib.git
            cd enlib
            export ENLIB_COMP=cca_intel
            make array_ops
        )
        (
            cd ${software_dir}
            git clone git@github.com:simonsobs/sofind.git
            cd sofind
            python -m pip install -e .
        )
        (
            cd ${software_dir}
            git clone git@github.com:simonsobs/mnms.git
            cd mnms
            python -m pip install -e .

        )
    fi

    export SOFIND_SYSTEM=perlmutter
    export PYTHONPATH=$PYTHONPATH:${software_dir}
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

First we need to create all the window functions. In the following we will assume that the window functions  used in temperature and in polarisation are the same, we will create the windows based on a the edges of the survey, a galactic mask, a pt source mask and a threshold on the amount of crosslinking in the patch of observation.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_window_dr6.py global_dr6_v4.dict
    # real	10m2.348s

The next step is to precompute the mode coupling matrices associated with these window functions, we have N window functions corresponding to each (season X array a) data set, we will have to compute all the cross power spectra of the form
(season X array 1)  x (season Y array 2) there are therefore Ns = N * (N+1)/2 independent spectra to compute

.. code:: shell

    salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_mcm_and_bbl.py global_dr6_v4.dict
    # real 23m10.708s

Now we can compute all the power spectra, the mpi loop is done on all the different arrays.
If you consider five detector arrays, we first compute the alms using mpi, and then have a simple code to combine them into power spectra

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_alms.py global_dr6_v4.dict
    # real	3m47.856s
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_spectra_from_alms.py global_dr6_v4.dict
    # real	7m6.917s


Finally, we need to compute the associated covariances of all these spectra, for this we need a model for the signal and noise power spectra

.. code:: shell

    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_best_fit_mflike.py global_dr6_v4.dict
    # real	0m42.667s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_noise_model.py global_dr6_v4.dict
    # real	0m40.229s

The computation of the covariance matrices is then divided into two steps, first compute all (window1 x window2) alms needed for the covariance computation, then the actual computation, note that there is Ns(Ns+1)/2 covariance matrix block to compute, this is enormous and is therefore the bottleneck of the spectra computation.


.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=36 srun -n 7 -c 36 --cpu-bind=cores python get_sq_windows_alms.py global_dr6_v4.dict
    # real 0m31.524s
    salloc --nodes 2 --qos interactive --time 03:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 8 -c 64 --cpu-bind=cores python get_covariance_blocks.py global_dr6_v4.dict
    # real	89m7.793s

you might also want to compute the beam covariance

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_beam_covariance.py global_dr6_v4.dict
    # real 3m56.972s

Now you might want to combine the spectra together (although it might be a bit early as we will explained later), in any case the code to do the combination is the following

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_xarrays_covmat.py global_dr6_v4.dict
    # real 1m20.820s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_xfreq_spectra.py global_dr6_v4.dict
    # real 2m16.029s

So why was it early, well the spectra are contaminated by leakage, and the analytic covariance computation might under estimate the errorbars, in order to correct for leakage go in the leakage folder

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_leakage_corrected_spectra.py global_dr6_v4.dict
    # real 1m4.582s
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_leakage_sim.py global_dr6_v4.dict
    # real 15m50.472s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_leakage_covariance.py global_dr6_v4.dict
    # real 6m38.858s

To generate a set of simulated spectra using the `mnms` noise simulation code you first have to generate the noise `alms` for each split and wafer and store them to disk. Then you have to run a standard simulation routine that reads the precomputed noise `alms`. Remember to delete the noise `alms` when you are done with your simulations. For a set of 100 simulations :

.. code:: shell

    salloc --nodes 2 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=128 srun -n 4 -c 128 --cpu_bind=cores python mc_mnms_get_nlms.py global_dr6_v4.dict
    # real time ~ 3h (for 100 sims)

    salloc --nodes 4 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu_bind=cores python mc_mnms_get_spectra_from_nlms.py global_dr6_v4.dict
    # real time ~ 1100s for each sim


To estimate the kspace filter transfer function from simulations

.. code:: shell

    salloc --nodes 4 --qos interactive --time 3:00:00 --constraint cpu
    OMP_NUM_THREADS=32 srun -n 32 -c 32 --cpu-bind=cores python mc_get_kspace_tf_spectra.py global_dr6_v4.dict

    salloc --nodes 1 --qos interactive --time 1:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_kspace_tf_analysis.py global_dr6_v4.dict
