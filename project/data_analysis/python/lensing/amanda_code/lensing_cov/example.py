"""
From Amanda MacInnis: calculate and save the blocks of the cmb + lensing covmat
Note that you would have to have precompted the derivatives with CLASS delens
see: https://docs.google.com/document/d/1CqTPxw3_Tlrzd8iQ_RDgTuSUSkeeDJDRJduFRNbHsAQ/edit#heading=h.6lqm93l8zanl
"""
import os
import numpy as np
from lensing_covariance import LensingCovariance


# directory where CLASS delens output was saved, and the root used to save it:
class_delens_root = './derivatives/lmax8500_00'
# which blocks and terms to calculate
use_derivs_wrt_lensing = True   # CMB x CMB term
use_derivs_wrt_unlensed = True # CMB x CMB term
calc_lensingxcmb = True # CMB x lensing blocks; NOTE that you need the derivatives of CMB w/ respect to lensing spectrum for this

# where to save the output, and what the file names will begin with:
output_root = 'lmax8500_'

# other inputs :
fsky = 0.257
cmb_spectra = ['tt', 'te', 'ee', 'bb'] # list of CMB spectra to use; C_L^kk is also included automatically if `calc_lensingxcmb = True`.

# set the multipole ranges for CMB (all the same here) and for C_L^kk:
lmin = 2
lmax = 8500
Lmin = 2
Lmax = 8500
# note that you can choose a different `lmin` and `lmax` for each individual spectrum:
ell_ranges = {'tt': [lmin, lmax], 'te': [lmin, lmax], 'ee': [lmin, lmax], 'bb': [lmin, lmax], 'kk': [Lmin, Lmax]} 

# binning matrices for lensed CMB C_l's and for C_L^kk:
cmb_bmat_file = 'cmb_bmat_lmin2_lmax10000.txt'
lens_bmat_file = 'lensing_bmat_Lmin2_Lmax4000.txt'
cmb_bmat = np.loadtxt(cmb_bmat_file)
lens_bmat = np.loadtxt(lens_bmat_file)

bmat_lmin = 2 # minimum multipole in spectra/covmat to be binned
cmb_bin_ranges = [7, None] # keep these bins for the CMB; `None` means we keep all bins above `bmin`
lens_bin_ranges = [1, 47] # keep these bins for C_L^kk
bin_ranges = {s: cmb_bin_ranges for s in cmb_spectra}
bin_ranges['kk'] = lens_bin_ranges

# load the unlensed CMB (C_ell in uK^2) + lensing convergence (C_L^kk) theory spectra into a dictionary: 
theory_file = 'theory_cls.txt'
theory_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk'] # order of columns in theory file
theory = np.loadtxt(theory_file)
theo = {} 
for i, col in enumerate(theory_cols):
    theo[col] = theory[:,i][:8501]

# pass everything to the `LensingCovariance` class:
lcov = LensingCovariance(theo, ell_ranges, class_delens_root, fsky=fsky, 
        cmb_bmat=cmb_bmat, lens_bmat=lens_bmat, bmat_lmin=bmat_lmin, bin_ranges=bin_ranges, 
        use_derivs_wrt_lensing=use_derivs_wrt_lensing, use_derivs_wrt_unlensed=use_derivs_wrt_unlensed, calc_lensingxcmb=calc_lensingxcmb)

# calculate, bin, and save the blocks of the covariance matrix:
binned_blocks = lcov.cov_blocks(binned=False, save=True) # returns a nested dict, e.g. kk x TT covariance is `binned_blocks['kk']['tt']`









