**************************
INSTALLING THE ACT POWER SPECTRUM PIPELINE AT NERSC
**************************

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

Requirements
============

* pspy >= 1.7.0
* pspipe_utils >= 0.1.4


Running the dr6 main analysis
------------------
To run the main dr6 analysis follow the instruction in `dr6 <https://github.com/simonsobs/PSpipe/tree/master/project/data_analysis/dr6.rst/>`_

Running the dr6xPlanck pipeline
------------------
To run the dr6xPlanck analysis follow the instruction in `dr6xplanck <https://github.com/simonsobs/PSpipe/tree/master/project/data_analysis/dr6xplanck.rst/>`_

Estimation of the dust
------------------
To estimate the dust in the dr6 patch, we use Planck 353 GHz maps  <https://github.com/simonsobs/PSpipe/tree/master/project/data_analysis/dust.rst/>`_

Running our reproduction of the Planck pipeline
------------------
To run a reproduction of the Planck official result follow the instruction in `planck <https://github.com/simonsobs/PSpipe/tree/master/project/data_analysis/planck.rst/>`_

kspace filter TF and aberration correction
------------------
Some extra instruction to compute the kspace filter tranfer function and the aberration correction follow the instruction in `planck <https://github.com/simonsobs/PSpipe/tree/master/project/data_analysis/xtra.rst/>`_
