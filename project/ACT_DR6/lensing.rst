**************************
Non gaussian lensing term
**************************

Here are some specific instructions to compute the contribution due to lensing to the covariance matrix
There are three terms to consider
- two standard non gaussian contributions from lensing: https://arxiv.org/pdf/1205.0474.pdf
- the super sample lensing covariance based on this paper: https://arxiv.org/abs/1401.7992

in order to compute both, we are following instruction from Amanda MacInnis:


Standard lensing term
------------------

we will use the code in the lensing/amanda_code/lensing_cov folder
- Download and compile CLASS DELENS https://github.com/selimhotinli/class_delens (make class)
- Run it with the following `paramfile: <https://github.com/simonsobs/PSpipe/blob/master/project/data_analysis/python/lensing/amanda_code/lensing_cov/class_delens_template.ini/>`_ (this will take 2 hours on nersc for this lmax)
- this will create a bunch of file, create a folder called derivatives and copy them there, then run example.py, it will use the derivatives to create the two standard non gaussian contributions from lensing


Super sample lensing
------------------
we will use the code in the lensing/amanda_code/lensing_super_sample_cov folder
- Run python ssc_example.py


