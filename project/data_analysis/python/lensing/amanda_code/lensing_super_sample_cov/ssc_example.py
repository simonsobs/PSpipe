"""
from Amanda MacInnis: calculate and save the SSC term in the blocks of the covmat
"""
import os
import numpy as np
from ssc_covariance import calculate_ssc_blocks
from pixell import enmap


# where to save the output, and what the file names will begin with:
output_root = 'test_ssc'

cmb_spectra = ['tt', 'te', 'ee', 'bb'] # list of CMB spectra to use

# set the multipole ranges for CMB (all the same here) and for C_L^kk:
lmin = 2
lmax = 8500
Lmin = 2
Lmax = 8500 # used in sigma_kappa^2 calculation (eq. 3 of arXiv:1401.7992)
# note that you can choose a different `lmin` and `lmax` for each individual spectrum:
ell_ranges = {'tt': [lmin, lmax], 'te': [lmin, lmax], 'ee': [lmin, lmax], 'bb': [lmin, lmax], 'kk': [Lmin, Lmax]} 

# the mask (including point source masks, etc.) 
mask_fname = '/pscratch/sd/t/tlouis/data_analysis_v4_march24/final/dr6/windows/window_dr6_pa5_f090_baseline.fits'

# file holding the LENSED CMB theory and CMB lensing convergence spectrum
lensed_theory_file = 'lensed_theory_cls.txt'
theory_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk'] # order of columns in theory file


# load inputs: mask, and lensed CMB (C_ell in uK^2) + lensing convergence (C_L^kk) theory spectra
lensed_theory = np.loadtxt(lensed_theory_file)
lensed_theo = {}
for i, col in enumerate(theory_cols):
    lensed_theo[col] = lensed_theory[:,i]
mask = enmap.read_map(mask_fname)

# calculate the SSC term for the different cov. blocks (TT x TT, TT x EE, etc.)
print('calculating SSC')
ssc_blocks = calculate_ssc_blocks(mask, lensed_theo, ell_ranges, Lmax, cmb_spectra=cmb_spectra)

# save them:
if not os.path.isdir(output_root):
    output_root = f'{output_root}_'
for i, s1 in enumerate(cmb_spectra):
    for s2 in cmb_spectra[i:]:
        fname = f'{output_root}unbinned_{s1}x{s2}.npy'
        print(f'saving {s1} x {s2} SSC term to {fname}')
        np.save(fname, ssc_blocks[s1][s2])

