**************************
Estimating the dust using Planck 353 GHZ
**************************

for this you would need the window to be pre-computed, use the ones computed during the AxP run, so that they contain at 12 arcmin source mask.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 3 -c 64 --cpu-bind=cores python get_mcm_and_bbl.py global_dust.dict
    # real   0m52.889s
    OMP_NUM_THREADS=128 srun -n 2 -c 128 --cpu-bind=cores python get_alms.py global_dust.dict
    # real   1m11.137s
    OMP_NUM_THREADS=64 srun -n 3 -c 64 --cpu-bind=cores python get_spectra_from_alms.py global_dust.dict
    # real   0m16.655s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_best_fit_mflike.py global_dust.dict
    # real   0m56.084s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_noise_model.py global_dust.dict
    # real   0m28.121s
    OMP_NUM_THREADS=64 srun -n 3 -c 64 --cpu-bind=cores python get_sq_windows_alms.py global_dust.dict
    OMP_NUM_THREADS=42 srun -n 6 -c 42 --cpu-bind=cores python get_covariance_blocks.py global_dust.dict
    # real   0m58.895s

you can run simulations for 353 GHz and 143 GHz using the code in the planck : get_planck_sim_nlms.py, and the one in the montecarlo folder: mc_mnms_get_spectra_from_nlms.py.
You can analyse and plot the simulation results using

.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_analysis.py global_dr6_v4.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_cov_analysis.py global_dr6_v4.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_plot_spectra.py global_dr6_v4.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_plot_covariances.py global_dr6_v4.dict

You can then fit the dust amplitude using

.. code:: shell

    salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu
    
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python fit_dust_amplitude.py global_dust.dict --mode TT
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python fit_dust_amplitude.py global_dust.dict --mode TE
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python fit_dust_amplitude.py global_dust.dict --mode TB
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python fit_dust_amplitude.py global_dust.dict --mode EE
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python fit_dust_amplitude.py global_dust.dict --mode BB
