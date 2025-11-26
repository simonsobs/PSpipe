from pspipe_utils import consistency, external_data, pspipe_list, covariance
from pspy import so_spectra, so_cov, pspy_utils
import numpy as np
import pylab as plt

def get_residual_and_cov(map_set_list, spec_dir, cov_dir, mode, spectra_order, op="aa+bb-2ab", mc_cov=False):
    """
    get the corresponding residual and its associated covariance for the given map_set_list,
    the given mode and the given operation
    
    Parameters
    ----------
    map_set_list : list of string
        the map_set we want to compare, for example ["Planck_f143", "Planck_f353"]
    spec_dir: string
        the location of the spectra
    cov_dir: string
        the location of the covariances
    mode: string
        the spectrum we want to look at, e.g "TT"
    spectra_order: string
        the order of the spectra, e.g ["TT", "TE", "TB", ..., "BB"]
    op: string
        the operation we want to perform
    mc_cov: boolean
        wether to use a monte carlo correction for the covariance
    
    """
    
    ps_template = spec_dir + "/Dl_{}x{}_cross.dat"
    cov_template = cov_dir + "/analytic_cov_{}x{}_{}x{}.npy"
    ps_dict, cov_dict = consistency.get_ps_and_cov_dict(map_set_list, ps_template, cov_template, spectra_order=spectra_order)
    lb, res, cov_res, _, _ = consistency.compare_spectra(map_set_list, op, ps_dict, cov_dict, mode=mode)
    
    if mc_cov:
        mc_cov_template = cov_dir + "/mc_cov_{}x{}_{}x{}.npy"
        _, mc_cov_dict = consistency.get_ps_and_cov_dict(map_set_list, ps_template, mc_cov_template, spectra_order=spectra_order)
        _, _, mc_cov_res, _, _ = consistency.compare_spectra(map_set_list, op, ps_dict, mc_cov_dict, mode=mode)
        cov_res = covariance.correct_analytical_cov(cov_res,
                                                    mc_cov_res,
                                                    only_diag_corrections=True)

    return lb, res, cov_res


def get_spectra_and_cov(spec_dir, cov_dir, spec_name, mode, spectra_order, mc_cov=False):
    """
    get the corresponding spectrum and its associated covariance for the given spec_name,
    the given mode
    
    Parameters
    ----------
    spec_dir: string
        the location of the spectra
    cov_dir: string
        the location of the covariances
    spec_name: string
        the name of the spectrum to look at, e.g "dr6_pa4_f220xdr6_pa4_f220"
    mode: string
        the spectrum we want to look at, e.g "TT"
    spectra_order: string
        the order of the spectra, e.g ["TT", "TE", "TB", ..., "BB"]
    mc_cov: boolean
        wether to use a monte carlo correction for the covariance
    
    """


    lb, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{spec_name}_cross.dat", spectra=spectra_order)
    cov = np.load(f"{cov_dir}/analytic_cov_{spec_name}_{spec_name}.npy")
    cov = so_cov.selectblock(cov, spectra_order, n_bins=len(lb), block=mode+mode)
    if mc_cov:
        mc_cov = np.load(f"{cov_dir}/mc_cov_{spec_name}_{spec_name}.npy")
        mc_cov = so_cov.selectblock(cov, spectra_order, n_bins=len(lb), block=mode+mode)
        cov = covariance.correct_analytical_cov(cov,
                                                mc_cov,
                                                only_diag_corrections=True)

    
    return lb, ps[mode], cov


def load_band_pass(dict, use_220=False):
    "load the bandpass corresponding to the dictionnary, optionnaly also read the 220 GHz bandpass"

    narrays, sv_list, ar_list = pspipe_list.get_arrays_list(dict)

    passbands = {}
    for sv, ar in zip(sv_list, ar_list):
        freq_info = dict[f"freq_info_{sv}_{ar}"]

        if dict["do_bandpass_integration"]:
            nu_ghz, pb = np.loadtxt(freq_info["passband"]).T
        else:
            nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.])

        passbands[f"{sv}_{ar}"] = [nu_ghz, pb]

    if use_220:
        if dict["do_bandpass_integration"]:
            passbands[f"dr6_pa4_f220"] = external_data.get_passband_dict_dr6(["pa4_f220"])["pa4_f220"]
        else:
            passbands[f"dr6_pa4_f220"] = [[220], [1.0]]
            
    return passbands
    


