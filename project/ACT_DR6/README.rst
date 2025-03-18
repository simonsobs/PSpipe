***************************************************
INSTALLING THE ACT POWER SPECTRUM PIPELINE AT NERSC
***************************************************

Here we give instructions to install and to run the full thing on interactive nodes, you can also submit it to NERSC standard nodes

Installation steps
------------------

First, we strongly recommand to install everything in a virtual ``python`` environment in order to
avoid clash with other ``python`` modules installed, for instance, within the ``.local``
directory. You can use the `setup.sh
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/setup.sh>`_ script file to
install everything (the ``mpi4py`` installation command is especially important @ NERSC).

.. code:: shell

    source setup.sh

In this case, everything will be installed in the current working directory. You can set the
installation path by exporting the ``BASE_DIR`` before running the ``source`` command.

Every time you log to NERSC machines, you **need to source this file** with ``source setup.sh`` to
get into the virtual environment and use the proper software suite.

Running the DR6 pipelines
-------------------------

When installing ``pspipe``, you will get a ``pspipe-run`` binary that can be used to sequentially
run the different modules involved in DR6 power spectra production. The ``pspipe-run`` is feed by a
``yaml`` file that holds the sequence of modules with their corresponding computer resources needs @
NERSC. For instance, you can run the DR6 pipeline (see next section) with the following command

.. code:: shell

    pspipe-run -p ACT_DR6/yaml/pipeline_dr6.yml

Within the ``yaml/pipeline_dr6.yml`` file, the pipeline itself is defined inside the ``pipeline``
block where a module block is defined as follow

.. code:: yaml

    get_covariance_blocks:
      force: true
      minimal_needed_time: 03:00:00
      slurm:
        nodes: 4
        ntasks: 16
        cpus_per_task: 64

The module name refers to the ``python`` script located in ``data_analysis/python``
directory. Another script directory can be set on top of the ``yaml`` file with the
``script_base_dir`` variable. The ``force: true`` directive means the module will always be
processed even if it was already done. The other parameters relate to slurm allocation when running
the pipeline in an **interactive node**.

If you want to use the pipeline in batch mode, you can refer to `pipeline_mnms.yml
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/yaml/pipeline_mnms.yml>`_ file to
setup the slurm batch allocation. Then, you can send the job by doing

.. code:: shell

    pspipe-run -p data_analysis/yaml/pipeline_mnms.yml --batch


This will add the job to the slurm queue and you can monitor your job with the usual ``squeue``
command.

Another useful example of batch job is the possibility to run Monte Carlo Markov Chain with
``cobaya``. The file `pipeline_mcmc.yml
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/yaml/pipeline_mcmc.yml>`_ shows how
the likelihood is prepared with the ``sacc`` file produced by the pipeline and how the chains are
setup for ``cobaya`` with ``mpi`` support.

The next sections will be linked to their corresponding ``pipeline.yml`` file.

Running the dr6 main analysis
-----------------------------

You can  run the whole pipeline with the `pipeline_dr6.yml
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/yaml/pipeline_dr6.yml>`_ file.
Detailed instructions for what the pipeline does are in `dr6
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/dr6.rst>`_.
Once the pipeline has been ran, and the best fit cosmology has been obtained, you can reproduce the paper figures following the instructions in
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/dr6_paper_plot.rst>`_ .


Running the dr6xPlanck pipeline
-------------------------------
You can  run the whole pipeline of ACT DR6 x legacy with the  `pipeline_dr6xlegacy.yml
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/yaml/pipeline_dr6xlegacy.yml>`_ file.
Detailed instructions for what the pipeline does are in `dr6xplanck
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/dr6xplanck.rst>`_.


Estimation of the dust
----------------------
You can  run the whole pipeline with the `pipeline_dust.yml
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/yaml/pipeline_dust.yml>`_ file.
Detailed instructions for what the pipeline does are in `dust
<https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/dust.rst/>`_.

