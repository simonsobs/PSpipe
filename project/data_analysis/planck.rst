**************************
Reproducing planck official result
**************************

Here are some specific instructions to reproduce Planck official spectra .
There are two different dictionnary files ``planck/global_legacy.dict`` and ``planck/global_NPIPE.dict``
for compute legacy spectra and Npipe spectra respectively.
Unlike ACT, Planck uses different beam/window for different splits so you will use the code in the ``per_split`` folder
We compute the spectra in the official Planck legacy window function


.. code:: shell

    salloc -N 1 -C cpu -q interactive -t 02:00:00
    OMP_NUM_THREADS=32 srun -n 6 -c 32 --cpu_bind=cores python get_mcm_and_bbl_per_split.py global_legacy.dict
    OMP_NUM_THREADS=32 srun -n 8 -c 32 --cpu_bind=cores python get_alms_per_split.py global_legacy.dict
    OMP_NUM_THREADS=32 srun -n 8 -c 32 --cpu_bind=cores python get_spectra_from_alms_per_split.py global_legacy.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_best_fit_mflike.py global_legacy.dict

to get the covariance we will use simulation only  :

.. code:: shell

    salloc -N 4 -C cpu -q interactive -t 02:00:00
    OMP_NUM_THREADS=4 srun -n 256 -c 4 --cpu_bind=cores python get_planck_sim_nlms.py global_legacy.dict
    # real    21m47.004s
    OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu_bind=cores python mc_mnms_get_spectra_from_nlms_per_split.py global_legacy.dict


