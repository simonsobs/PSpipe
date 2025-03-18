**************************
Figures for the paper
**************************

After running the dr6 pipeline, and after the bestfit cosmology has been obtained, you can run the scripts producing the different figures of the paper.
They are located in the  "postlikelihood" and "paper_plots" folders.

First we calibrate the data and combine the different spectra together

.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_best_fit_mflike.py post_likelihood_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python apply_likelihood_calibration.py post_likelihood_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python get_combined_spectra.py post_likelihood_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python results_plot_combined_spectra.py post_likelihood_bin50.dict


You can also produce the files we have released on lambda by running:

.. code:: shell

    salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python spectra_release.py post_likelihood_bin50.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python plot_spectra_release.py post_likelihood_bin50.dict

To plot the ACT DR6 spectra together with planck run

.. code:: shell

    python results_plot_with_planck.py post_likelihood_bin50.dict
    
    
To plot the residuals with respect to a LCDM model use either

.. code:: shell

    python results_plot_residuals.py post_likelihood_bin50.dict
    python results_plot_residuals.py post_likelihood_PACT_bin50.dict


PACT is for the bestfit model of ACT + Planck.

To plot the x-array TT power spectra together with their best fit model use

.. code:: shell

    python results_multifrequency_components_TT.py post_likelihood_bin50.dict

To plot the x-freq TT power spectra

.. code:: shell

    python results_multifrequency_TT.py post_likelihood_bin50.dict

To plot the x-freq EE and TE power spectra together with their null residuals

.. code:: shell

    python results_get_combined_null.py post_likelihood_bin50.dict
    python results_get_combined_null_TEEE.py post_likelihood_bin50.dict

To redo the plot showing how the tSZ power spectrum depends on alpha_tSZ

.. code:: shell

    python results_tsz_plot.py post_likelihood_bin50.dict
    
To redo the plot showing the dust contamination

.. code:: shell
    python results_plot_dust.py post_likelihood_bin50.dict
    
To redo the plot showing the relative contribution of each covariance term

.. code:: shell
    python results_compare_covariance_terms.py post_likelihood_bin50.dict

To redo the plot showing the comparison of error for ACT and Planck

.. code:: shell
    python results_compare_ACT_Planck_errors.py post_likelihood_bin50.dict

To redo the plot of the B modes power spectrum and estimation of its amplitude grab the code in "paper_plots/B_modes" and run

.. code:: shell
    python results_BB_likelihood.py post_likelihood_bin50.dict

Then you can redo the fit for polarisation angle with

.. code:: shell
    python results_EB_likelihood.py post_likelihood_bin50.dict
    python results_plot_pol_angle.py post_likelihood_bin50.dict

You can plot various noise properties with

.. code:: shell
    python results_noise_spectrum.py post_likelihood_bin50.dict
    python results_noise_spectrum_all.py post_likelihood_bin50.dict
    python results_noise_spectrum_correlation.py post_likelihood_bin50.dict


