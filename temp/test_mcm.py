from pspy import so_map,sph_tools
import so_mcm
import healpy as hp, pylab as plt, numpy as np

win=so_map.read_map('window.fits')
#win.plot()
l,bl=np.loadtxt('../data/beam.txt',unpack=True)
lmax=3000
wlm= sph_tools.map2alm(win,niter=0,lmax=lmax)
binning_file='../data/BIN_ACTPOL_50_4_SC_low_ell_startAt2'
mbb, Bbl= so_mcm.mcm_and_bbl_spin0_thibaut(wlm,binning_file,lmax,bl1=bl,type='Dl')

