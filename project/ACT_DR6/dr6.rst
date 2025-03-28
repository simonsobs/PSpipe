**************************
Computing DR6 power spectra spectra
**************************

Here are some specific instructions to compute spectra for DR6 at NERSC.
Since it involves a large amount of spectrum computation, we are going to make full use of MPI capacities.
The current dictionary is called ``global_dr6v4_bin50.dict`` and is located in the ``paramfiles`` folder.
Then, we can use the codes in the ``python`` folder to run the pipeline sequentially.
Here, we provide instructions to install and run the full process on interactive nodes. You can, of course, also submit it to NERSC standard nodes.



Running the main pipeline
-------------------------------------------------------

First, we need to create all the window functions. In the following, we will assume that the window functions used in temperature and polarization are the same. We will create the windows based on the edges of the survey, a galactic mask, a point source mask, and a threshold on the amount of crosslinking in the observed patch.



.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_window_dr6.py global_dr6v4_bin50.dict
    # real	10m2.348s

The next step is to precompute the mode coupling matrices associated with these window functions, we have N window functions corresponding to each (array-bands) dataset, we will have to compute all the cross power spectra of the form
(array 1)  x (array 2) there are therefore Ns = N x (N + 1) / 2 = 15 independent set of spectra (TT - TE - TB - ET - BT - EE - EB - BE - BB) to compute, note that for array 1 = array 2, TE = ET, TB = BT, EB = BE

.. code:: shell

    salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_mcm_and_bbl.py global_dr6v4_bin50.dict
    # real 23m10.708s

Now we can compute all the power spectra. The MPI loop runs over all the different arrays.
If you consider five detector arrays, we first compute the alms  using MPI and then use a simple code to combine them into power spectra.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_alms.py global_dr6v4_bin50.dict
    # real	3m47.856s
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python get_spectra_from_alms.py global_dr6v4_bin50.dict
    # real	7m6.917s

Finally, we need to compute the associated covariances of all these spectra. For this, we need a model for the signal and noise power spectra.


.. code:: shell

    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_best_fit_mflike.py global_dr6v4_bin50.dict
    # real	0m42.667s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_noise_model.py global_dr6v4_bin50.dict
    # real	0m40.229s


The computation of the covariance matrices is then divided into two steps: first, compute all (window1 x window2) alms  needed for the covariance computation, then perform the actual computation. Note that there are Ns x (Ns + 1) / 2 = 120 covariance matrix blocks to compute. This is enormous and therefore represents the bottleneck of the spectra computation.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=36 srun -n 7 -c 36 --cpu-bind=cores python get_sq_windows_alms.py global_dr6v4_bin50.dict
    # real 0m31.524s
    salloc --nodes 4 --qos interactive --time 03:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu-bind=cores python get_covariance_blocks.py global_dr6v4_bin50.dict
    # real	89m7.793s
    
The beams have associated uncertainties that need to be propagated in the pipeline. To produce all associated beam covariance matrices, run:

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_beam_covariance.py global_dr6v4_bin50.dict
    # real 3m56.972s

Now, you might want to combine the different covariance matrix blocks to form a cross-array covariance matrix.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_xarrays_covmat.py global_dr6v4_bin50.dict
    # real 1m20.820s


So, this produces all of the main products: spectra and covariances. Now, we need to take into account some extra physical effects and systematics.


Leakage correction and leakage covariance
-------------------------------------------------------

The spectra are contaminated by leakage. In order to correct for leakage, you should grab the code in the leakage folder and run:

.. code:: shell

    salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_leakage_corrected_spectra_per_split.py global_dr6v4_bin50.dict
    # real 1m4.582s
    OMP_NUM_THREADS=12 srun -n 20 -c 12 --cpu-bind=cores python get_leakage_sim.py global_dr6v4_bin50.dict
    # real 15m50.472s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_leakage_covariance.py global_dr6v4_bin50.dict
    # real 6m38.858s
    

Monte-Carlo kspace filter transfer function
-------------------------------------------------------

To compute the Monte Carlo transfer function that encodes the k-space filter effect, you will grab the code in the kspace folder.


.. code:: shell

    salloc --nodes 4 --qos interactive --time 3:00:00 --constraint cpu
    OMP_NUM_THREADS=32 srun -n 32 -c 32 --cpu-bind=cores python mc_get_kspace_tf_spectra.py global_dr6v4_bin50.dict
    salloc --nodes 1 --qos interactive --time 1:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_kspace_tf_analysis.py global_dr6v4_bin50.dict

    
    
    
Monte Carlo correction to the covariance matrix
-------------------------------------------------------

To generate a set of simulated spectra using the `mnms` noise simulation code you first have to generate the noise `alms` for each split and wafer and store them to disk. Then, you need to run a standard simulation routine that reads the precomputed noise `alms`. Remember to delete the noise `alms` when you are done with your simulations. For a set of 80 simulations, grab the code in the montecarlo folder.

.. code:: shell

    salloc --nodes 2 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=128 srun -n 4 -c 128 --cpu_bind=cores python mc_mnms_get_nlms.py global_dr6v4_bin50.dict
    # real time ~ 4h (for 80 sims)

    salloc --nodes 4 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu_bind=cores python mc_mnms_get_spectra_from_nlms.py global_dr6v4_bin50.dict
    # real time ~ 1100s for each sim
    
You can analyze and plot the simulation results using:

.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_analysis.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_cov_analysis.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_plot_spectra.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_plot_covariances.py global_dr6v4_bin50.dict

In addition, if you wish to create a covariance matrix corrected from simulations using Gaussian processes, run:


.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python mc_cov_analysis_for_gp.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python get_mc_corrected_xarrays_covmat_gp.py global_dr6v4_bin50.dict




Aberration correction
-------------------------------------------------------
The spectra are aberrated, and we need to correct for it. To do so, we generate simulations with aberration and compare them with simulations without aberration. We then correct the effect on the data power spectra.
Grab the code in the aberration folder and run:

.. code:: shell

    salloc --nodes 4 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu_bind=cores python mc_get_aberrated_spectra.py global_dr6v4_bin50.dict
    # real time 94m56.700s for 100 sims

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python mc_aberration_analysis.py global_dr6v4_bin50.dict
    # real    2m31.819s
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_aberration_corrected_spectra.py global_dr6v4_bin50.dict
    # real    1m53.833s


Radio and tSZ trispectrum
-------------------------------------------------------

To include the non-Gaussian contribution to the covariance matrix coming from the connected four-point function of the Radio sources, CIB, and tSZ (assumed to be Poisson distributed), grab the code in the non_gaussian_fg folder and run:

.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_connected_trispectrum_radio.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_connected_trispectrum_tSZ.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_connected_trispectrum_CIB.py global_dr6v4_bin50.dict
    # real 3m4.125s
    
Non gaussian lensing terms
-------------------------------------------------------

To include the non gaussian contribution to the covariance matrix coming from the connected four point function due to lensing we rely on external codes (from Amanda MacInnis)
see the dedicated `README <https://github.com/simonsobs/PSpipe/tree/master/project/ACT_DR6/lensing.rst/>`_ for how these terms are computed.
Once you have ran amanda codes, run

.. code:: shell

    salloc --nodes 1 --qos interactive --time 1:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python lensing_combine_cov_mat.py global_dr6v4_bin50.dict

this will create all the blocks associated to the non lensing covariance term and a x_ar covariance matrix


We can check the analytic computation using PSpipe simulation code


.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python lensing_camb.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python lensing_gaussian_cov.py global_dr6v4_bin50.dict

    salloc --nodes 4 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 16 -c 64 --cpu_bind=cores python lensing_sim.py global_dr6v4_bin50.dict

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python lensing_analysis.py global_dr6v4_bin50.dict


Array null test
-------------------------------------------------------

To perform the array null test, grab the code in the null_tests folder and run:


.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python compute_null_tests.py global_dr6v4_bin50.dict


Combine cov mat and write data in a SACC file
-------------------------------------------------------

To finally combine all covariance matrices together and write the final data into a SACC file, run:


.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_combined_cov_mats.py global_dr6v4_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python port2sacc.py global_dr6v4_bin50.dict

