**************************
Figures for the paper
**************************

Here I assume that the entire pipeline described in `dr6 <https://github.com/simonsobs/PSpipe/tree/master/project/data_analysis/dr6.rst/>`_
has been ran, here is the script producing the different figures of the paper.


To plot the TT/TE/EE ACT DR6 power spectrum together with Planck public spectra

.. code:: shell

    python results_plot_with_planck.py post_likelihood.dict
    
To plot the residuals with respect to a LCDM model use either

.. code:: shell

    python results_plot_residuals.py post_likelihood.dict
    python results_plot_residuals.py post_likelihood_PACT.dict

PACT is for the bestfit model of ACT + Planck.

To plot the x-array TT power spectrum together with his best fit model and with its different components use

.. code:: shell

    python results_multifrequency_components_TT.py post_likelihood.dict

To plot the x-freq TT power spectrum

.. code:: shell

    python results_multifrequency_TT.py post_likelihood.dict

To plot the x-freq EE and TE power spectra together with their null residuals, you need to do some work.
First apply the systematic model to your simulation and combine your x-ar simulation spectra into x-freq cmb only simulation spectra

.. code:: shell

    python mc_apply_syst_model.py global_dr6.dict
    python mc_get_combined_spectra.py global_dr6.dict
    python mc_analyze_combined_spectra.py global_dr6.dict

Then you can make the plot

.. code:: shell

    python results_get_combined_null.py post_likelihood.dict

To redo the plot showing how the tSZ power spectrum depends on alpha_tSZ

.. code:: shell

    python results_tsz_plot.py post_likelihood.dict

To redo the plot showing the relative contribution of each covariance term

.. code:: shell

    python results_compare_covariance_terms.py post_likelihood.dict

To redo the plot showing the comparison of ACT and Planck errors (note that you do need the ACT-CMB only likelihood)

.. code:: shell

    python results_compare_ACT_Planck_errors.py post_likelihood.dict
