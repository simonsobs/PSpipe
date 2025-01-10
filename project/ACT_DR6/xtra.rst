**************************
kspace filter transfer function and aberration correction
**************************

To compute the transfer function that encodes the kspace filter effect you will run the code in the kspace folder

.. code:: shell

    salloc --nodes 4 --qos interactive --time 3:00:00 --constraint cpu
    OMP_NUM_THREADS=32 srun -n 32 -c 32 --cpu-bind=cores python mc_get_kspace_tf_spectra.py global_dr6_v4.dict

    salloc --nodes 1 --qos interactive --time 1:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_kspace_tf_analysis.py global_dr6_v4.dict
