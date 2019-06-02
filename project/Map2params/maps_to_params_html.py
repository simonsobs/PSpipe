from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
import  numpy as np, pylab as plt, healpy as hp
import os,sys
import so_noise_calculator_public_20180822 as noise_calc
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


plot_dir='plot'

spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

freqs=d['freqs']

