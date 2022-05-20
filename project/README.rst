********
Projects
********

This page describes the different projects we are working on in PSpipe, they are at different states of completion/documentation


data analysis 
=========== 

The goal of this project is to reproduce results for the Choi et al paper. 


maps2params 
=========== 

The goal of this project is to make a simplistic 'end-to-end' simulation analysis.
We start with a set of cosmological parameters, a foreground model and an instrument model.
We generate simulations of the sky for different frequency channels, convolve them with the beams of the instrument, and add to them experimental noise.
We compute the power spectra of the simulations given a galactic, survey a point source mask and compute their associated covariance matrices. 

Planck pspy
=============
This folder contains codes that can be used for the reproduction of the planck results using pspy.
It also contains different scripts generating and analysing simulations of the Planck data.


old/correlation_coeff 
=================

This folder contains the code used for this analysis: https://arxiv.org/abs/1909.03967 (now published in PRD).
We have computed the correlation coefficient R_TE of planck data.




old/analyse_sims 
=============
This folder contains the (old) code used for the analysis of the official SO simulations (from Andrea Zonca)


old/actpol 
=============
This folder contains some (old) code that can be used for the analysis of the actpol data
