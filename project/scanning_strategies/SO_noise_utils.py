import numpy as np
from pspy import pspy_utils

def HEALPIX_effective_fsky(window):
    data = window.data[:]
    npix = 12 * window.nside ** 2
    w2 = np.sum(data[data != 0] ** 2) / (npix)
    w4 = np.sum(data[data != 0] ** 4) / (npix)
    return  w2 ** 2 / w4

def CAR_effective_fsky(window):
    data = window.data[:]
    pixsize_map = data.pixsizemap()
    w2 = np.sum(data ** 2 * pixsize_map) / (4 * np.pi)
    w4 = np.sum(data ** 4 * pixsize_map) / (4 * np.pi)
    return  w2 ** 2 / w4



def quick_analytic_cov(l, Clth_dict, window, binning_file, lmax):

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_size)
    
    if window.pixel == "HEALPIX":
        fsky = HEALPIX_effective_fsky(window)
    elif window.pixel == "CAR":
        fsky = CAR_effective_fsky(window)

    
    prefac = 1 / ((2 * bin_c + 1) * fsky * bin_size)

    cov = {}
    cov["TT"] = Clth_dict["TaTc"] * Clth_dict["TbTd"]  + Clth_dict["TaTd"] * Clth_dict["TbTc"]
    cov["TE"] = Clth_dict["TaTc"] * Clth_dict["EbEd"]  + Clth_dict["TaEd"] * Clth_dict["EbTc"]
    cov["ET"] = Clth_dict["EaEc"] * Clth_dict["TbTd"]  + Clth_dict["EaTd"] * Clth_dict["TbEc"]
    cov["EE"] = Clth_dict["EaEc"] * Clth_dict["EbEd"]  + Clth_dict["EaEd"] * Clth_dict["EbEc"]
    
    mat_diag = np.zeros((4 * nbins, 4 * nbins))
    
    for count, spec in enumerate(["TT", "TE", "ET", "EE"]):
        lb, cov[spec] = pspy_utils.naive_binning(l, cov[spec], binning_file, lmax)
        cov[spec]  *= prefac
        for i in range(nbins):
            mat_diag[i + count * nbins, i + count * nbins] = cov[spec][i]
        
    return fsky, mat_diag
    
