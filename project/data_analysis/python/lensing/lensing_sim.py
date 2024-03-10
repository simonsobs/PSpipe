"""
Generate a bunch of gaussian and non gaussian simulation (signal only) and compute their power spectra
OMP_NUM_THREADS=16 srun -n 64 -c 16 --cpu-bind=cores python lensing_sim.py global_lensing.dict #2 min for doing 64 sim
"""
import numpy as np
import pylab as plt
from pixell import lensing, curvedsky
from pspy import pspy_utils, so_map, so_mcm, sph_tools, so_spectra, so_mpi, so_dict
from pspipe_utils import  get_data_path, log
import sys, time, copy


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

iStart = d["iStart"]
iStop = d["iStop"]
niter = d["niter"]
binned_mcm = d["binned_mcm"]
type = d["type"]
binning_file = d["binning_file"]
write_unbinned_spectra = True

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

lensing_dir = "lensing"
pspy_utils.create_directory(lensing_dir)


window = so_map.read_map(f"{lensing_dir}/window.fits")
shape, wcs = window.data.shape, window.data.wcs
lensed_map = so_map.car_template_from_shape_wcs(3, shape, wcs, dtype=np.float64)
    
lmax_spec = int(window.get_lmax_limit()) # max l from template pixellisation

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{lensing_dir}/", spin_pairs=spin_pairs)

ps_lensed_mat   = np.load(f"{lensing_dir}/ps_lensed_mat.npy")
ps_unlensed_mat = np.load(f"{lensing_dir}/ps_unlensed_mat.npy")

run_names = ["gaussian", "non_gaussian"]

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=iStart, imax=iStop)

for iii in subtasks:
    for run_name in run_names:
        t = time.time()

        if run_name == "gaussian":
            lensed_map.data[:] = curvedsky.rand_map((3,) + shape, wcs, ps_lensed_mat)
        if run_name == "non_gaussian":
            lensed_map.data[:], = lensing.rand_map((3,) + shape, wcs, ps_unlensed_mat,
                                                   lmax=d["lmax_sim"],
                                                   output="l",
                                                   verbose = False)

        alm =  sph_tools.get_alms(lensed_map, (window, window), niter, lmax_spec)
        l_, ps = so_spectra.get_spectra(alm, alm, spectra=spectra)
        
        if write_unbinned_spectra == True:
            ps_ = copy.deepcopy(ps)
            ll = np.arange(2, lmax_spec)
            cll = {f: ps_[f][ll] for f in spectra}
            ll, cll = so_spectra.deconvolve_mode_coupling_matrix(ll, cll, mbb_inv, spectra)
            for spec in spectra:
                cll[spec] *= (ll * (ll + 1) / (2 * np.pi))
            so_spectra.write_ps(f"{lensing_dir}/spectra_unbin_{run_name}_{iii:05d}.dat", ll, cll, type="Dl", spectra=spectra)


        lb, Db = so_spectra.bin_spectra(l_,
                                        ps,
                                        binning_file,
                                        lmax_spec,
                                        type=type,
                                        mbb_inv=mbb_inv,
                                        spectra=spectra,
                                        binned_mcm=binned_mcm)
                                        
        so_spectra.write_ps(f"{lensing_dir}/spectra_{run_name}_{iii:05d}.dat", lb, Db, type="Dl", spectra=spectra)
        log.info(f"Sim {iii} {run_name} took {time.time()-t} to compute")
