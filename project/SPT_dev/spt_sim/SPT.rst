**************************
Redoing SPT simulation analysis
**************************



SPT provides 500 unfiltered simulations and 500 filtered simulations, allowing us to estimate the transfer function.
Note that in addition to this 500 filtered simulations  (called masking_yes) there is 110 simulations called masking_no.
The masking yes simulations comprises the output maps produced with the masked timestream high-pass filter
(this was the setting used for the real-data timestreams). The second set corresponds to the output maps produced with
the masking turned off in the filtering process.
These maps are produced to study the effect of the masking, namely the “filtering artifacts” described in Section IV A 1 of C25 and Section III B 2 of Q26.



We have written script to do both 2d transfer function analysis

.. code:: shell

    salloc --nodes 4 --qos interactive --time 02:00:00 --constraint cpu
    OMP_NUM_THREADS=16 srun -n 16 -c 64 --cpu-bind=cores python spt_sim_get_tf2d.py global_spt.dict
    
and one dimensional transfer function analysis

.. code:: shell

    salloc --nodes 4 --qos interactive --time 04:00:00 --constraint cpu
    OMP_NUM_THREADS=8 srun -n 8 -c 128 --cpu-bind=cores python spt_sim_get_tf_spectra.py global_spt.dict


We have written three scripts to analyse these simulation


.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 1 -c 64 --cpu-bind=cores python spt_sim_recover_input_theory.py global_spt.dict

This script check that while analysing unfiltered simulations we recover the input theory used to generate the sims.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 1 -c 64 --cpu-bind=cores python spt_sim_recover_filtering_bias.py global_spt.dict

This script compares the masking_yes and masking_no simulation to reproduce the filtering artifacts, for now the reproduction fails, I believe this is because SPT inpaints the simulations when computing this term.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 1 -c 64 --cpu-bind=cores python spt_sim_recover_transfer_function.py global_spt.dict

This script compares filtered and unfiltered simulation to estimate the transfer function. It does it both for masking_yes and masking_no simulations, with and without alm filter.


