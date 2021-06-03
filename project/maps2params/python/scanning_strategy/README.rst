**************************
Scanning strategy
**************************

In this ``project`` we compute the noise and covariance properties corresponding to the different SO scanning strategy


Requirements
============

* pspy : python library for power spectrum estimation (https://github.com/simonsobs/pspy)


Pipeline flow chart
===================


Let's assume you run the script on NERSC, first create a folder and copy the code, then get access to a single node
(salloc -N 1 -C haswell -q interactive -t 02:00:00) and run


.. code:: shell

    python SO_copy_and_format_data.py global.dict
    python SO_noise_get_spectra.py global.dict
    python SO_noise_plot_spectra.py global.dict
    python SO_noise_get_covariance.py global.dict
    python SO_noise_plot_covariance.py global.dict
