"""
This script read the estimated power spectra of the simulation and apply the ACT DR6 systematic model
to propagate beam and leakage uncertainties it then write them to the folder  sim_spectra_syst
"""

import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_mpi
from pspipe_utils import pspipe_list, best_fits, log, leakage
import numpy as np
import pylab as plt
import sys

def get_beam_sim(mean, error_modes):
    """
    Generate a realisation of beam  from a mean value and the error modes
    
    Parameters:
    ----------
    mean: 1d array
        the mean value of the beam
    error_modes: 2d array
        the error modes corresponding to the beam measurement (lmax, nmodes)

    """
    n_modes = error_modes.shape[1]
    bl_sim = mean + error_modes @ np.random.randn(n_modes)
    return bl_sim

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
surveys = d["surveys"]
iStart = d["iStart"]
iStop = d["iStop"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]

bestfit_dir = "best_fits"
in_sim_spec_dir = d["sim_spec_dir"]
out_sim_spec_dir = "sim_spectra_syst"

pspy_utils.create_directory(out_sim_spec_dir)

bl_mean, bb_mean, error_m_beam, gamma, err_m_gamma, var = {}, {}, {}, {}, {}, {}

for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
        name = f"{sv}_{ar}"
        log.info(f"reading leakage info {name}")
        gamma[name], err_m_gamma[name], var[name] = {}, {}, {}

        l, gamma[name]["TE"], err_m_gamma[name]["TE"], gamma[name]["TB"], err_m_gamma[name]["TB"] = leakage.read_leakage_model(d[f"leakage_beam_{name}_TE"][0],
                                                                                                                               d[f"leakage_beam_{name}_TB"][0],
                                                                                                                               lmax,
                                                                                                                               lmin=2)


        cov = {}
        cov["TETE"] = leakage.error_modes_to_cov(err_m_gamma[name]["TE"])
        cov["TBTB"] = leakage.error_modes_to_cov(err_m_gamma[name]["TB"])

        var[name]["TETE"] = cov["TETE"].diagonal()
        var[name]["TBTB"] = cov["TBTB"].diagonal()
        var[name]["TETB"] = var[name]["TETE"] * 0

        beam_data = np.loadtxt(d[f"beam_T_{sv}_{ar}"]) #only do T for now T=P for ACT
        l_beam, bl_mean[name], error_m_beam[name]  = beam_data[2: lmax + 2, 0], beam_data[2: lmax + 2, 1], beam_data[2: lmax + 2, 2:]
        lb, bb_mean[name] = pspy_utils.naive_binning(l_beam, bl_mean[name], binning_file, lmax)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

ps_th_dict, leak_mean = {}, {}

for spec_name in spec_name_list:

    name1, name2 = spec_name.split("x")
    
    l_th, ps_th_dict[spec_name] = so_spectra.read_ps(f"{bestfit_dir}/cmb_and_fg_{spec_name}.dat", spectra=spectra)
    id = np.where(l_th < lmax)
    
    l_th = l_th[id]
    for spec in spectra:
        ps_th_dict[spec_name][spec] = ps_th_dict[spec_name][spec][id]

    l, leak_mean[spec_name] = leakage.leakage_correction(l_th,
                                                         ps_th_dict[spec_name],
                                                         gamma[name1],
                                                         var[name1],
                                                         lmax,
                                                         gamma_beta=gamma[name2],
                                                         binning_file=binning_file)


so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=iStart, imax=iStop)
for iii in subtasks:

    log.info(f"applying syst model on sim {iii:05d}")
    
    gamma_sim, bb_sim = {}, {}
    
    for name in gamma.keys():
        gamma_sim[name] = {}
        gamma_sim[name]["TE"] = leakage.leakage_beam_sim(gamma[name]["TE"], err_m_gamma[name]["TE"])
        gamma_sim[name]["TB"] = leakage.leakage_beam_sim(gamma[name]["TB"], err_m_gamma[name]["TB"])
        bl_sim = get_beam_sim(bl_mean[name], error_m_beam[name])
        lb, bb_sim[name] = pspy_utils.naive_binning(l_beam, bl_sim, binning_file, lmax)
        
    for spec_name in spec_name_list:
        
        name1, name2 = spec_name.split("x")

        spec_name_cross = f"{type}_{spec_name}_cross_{iii:05d}"
        
        lb, leak_sim = leakage.leakage_correction(l_th,
                                                  ps_th_dict[spec_name],
                                                  gamma_sim[name1],
                                                  var[name1],
                                                  lmax,
                                                  gamma_beta=gamma_sim[name2],
                                                  binning_file=binning_file)

        lb, Db = so_spectra.read_ps(in_sim_spec_dir + f"/{spec_name_cross}.dat", spectra=spectra)

        for spec in spectra:
            Db[spec] *= (bb_sim[name1] * bb_sim[name2]) / (bb_mean[name1] * bb_mean[name2])
            Db[spec] += leak_sim[spec] - leak_mean[spec_name][spec]
            
        so_spectra.write_ps(out_sim_spec_dir + f"/{spec_name_cross}.dat", lb, Db, type, spectra=spectra)
