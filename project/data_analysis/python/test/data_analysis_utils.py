"""
Some utility functions for the data analysis project.
"""
import numpy as np
import healpy as hp
import pylab as plt
import os
from pixell import curvedsky
from pspy import pspy_utils, so_cov, so_spectra, so_mcm, so_map_preprocessing
from pspy.cov_fortran.cov_fortran import cov_compute as cov_fortran
from pspy.mcm_fortran.mcm_fortran import mcm_compute as mcm_fortran
from pixell import enmap
import gc

def get_filtered_map(orig_map, binary, filter, inv_pixwin_lxly=None, weighted_filter=False, tol=1e-4, ref=0.9):

    """Filter the map in Fourier space using a predefined filter. Note that we mutliply the maps by a binary mask before
    doing this operation in order to remove pathological pixels
    We also include an option for removing the pixel window function

    Parameters
    ---------
    orig_map: ``so_map``
        the map to be filtered
    binary:  ``so_map``
        a binary mask removing pathological pixels
    filter: 2d array
        a filter applied in fourier space
    inv_pixwin_lxly: 2d array
        the inverse of the pixel window function in fourier space
    weighted_filter: boolean
        wether to use weighted filter a la sigurd
    tol, ref: floats
        only in use in the case of the weighted filter, these arg
        remove crazy pixels value in the weight applied

    """
    
    if weighted_filter == False:
        if inv_pixwin_lxly is not None:
            orig_map = fourier_mult(orig_map, binary, filter * inv_pixwin_lxly)
        else:
            orig_map = fourier_mult(orig_map, binary, filter)

    else:
        orig_map.data *= binary.data
        one_mf = (1 - filter)
        rhs    = enmap.ifft(one_mf * enmap.fft(orig_map.data, normalize=True), normalize=True).real
        gc.collect()
        div    = enmap.ifft(one_mf * enmap.fft(binary.data, normalize=True), normalize=True).real
        del one_mf
        gc.collect()
        div    = np.maximum(div, np.percentile(binary.data[::10, ::10], ref * 100) * tol)
        orig_map.data -= rhs / div
        del rhs
        del div
        gc.collect()
        
        if inv_pixwin_lxly is not None:
            ft = enmap.fft(orig_map.data, normalize=True)
            ft  *= inv_pixwin_lxly
            orig_map.data = enmap.ifft(ft, normalize=True).real

    gc.collect()
    return orig_map
    
def fourier_mult(orig_map, binary, fourier_array):

    """do a fourier multiplication of the FFT of the orig_map with a fourier array, binary help to remove pathological pixels

    Parameters
    ---------
    orig_map: ``so_map``
        the map to be filtered
    binary:  ``so_map``
        a binary mask removing pathological pixels
    fourier_array: 2d array
        the fourier array we want to multiply the FFT of the map with
    """
    orig_map.data *= binary.data
    ft = enmap.fft(orig_map.data, normalize=True)
    ft  *= fourier_array
    orig_map.data = enmap.ifft(ft, normalize=True).real
    
    return orig_map
    

def get_coadded_map(orig_map, coadd_map, coadd_mask):
    """Co-add a map with another map given its associated mask.

    Parameters
    ---------
    orig_map: ``so_map``
        the original map without point sources
    coadd_map: ``so_map``
        the map to be co-added
    coadd_mask: ``so_map``
        the mask associated to the coadd_map
    """
    if coadd_map.ncomp == 1:
        coadd_map.data *= coadd_mask.data
    else:
        coadd_map.data[:] *= coadd_mask.data
    orig_map.data += coadd_map.data

    return orig_map


def fill_sym_mat(mat):
    """Make a upper diagonal or lower diagonal matrix symmetric

    Parameters
    ----------
    mat : 2d array
      the matrix we want symmetric
    """
    return mat + mat.T - np.diag(mat.diagonal())

def get_nspec(dict):

    surveys = dict["surveys"]
    nspec = {}

    for kind in ["cross", "noise", "auto"]:
        nspec[kind] = 0
        for id_sv1, sv1 in enumerate(surveys):
            arrays_1 = dict["arrays_%s" % sv1]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = dict["arrays_%s" % sv2]
                    for id_ar2, ar2 in enumerate(arrays_2):

                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        if  (id_sv1 > id_sv2) : continue
                        if (sv1 != sv2) & (kind == "noise"): continue
                        if (sv1 != sv2) & (kind == "auto"): continue
                        nspec[kind] += 1
                    
    return nspec

def get_noise_matrix_spin0and2(noise_dir, survey, arrays, lmax, nsplits):
    
    """This function uses the measured noise power spectra
    and generate a three dimensional array of noise power spectra [n_arrays, n_arrays, lmax] for temperature
    and polarisation.
    The different entries ([i,j,:]) of the arrays contain the noise power spectra
    for the different array pairs.
    for example nl_array_t[0,0,:] =>  nl^{TT}_{ar_{0},ar_{0}),  nl_array_t[0,1,:] =>  nl^{TT}_{ar_{0},ar_{1})
    this allows to consider correlated noise between different arrays.
    
    Parameters
    ----------
    noise_data_dir : string
      the folder containing the noise power spectra
    survey : string
      the survey to consider
    arrays: 1d array of string
      the arrays we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
      nl_per_split= nl * n_{splits}
    """
    
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    n_arrays = len(arrays)
    nl_array_t = np.zeros((n_arrays, n_arrays, lmax))
    nl_array_pol = np.zeros((n_arrays, n_arrays, lmax))
    
    for c1, ar1 in enumerate(arrays):
        for c2, ar2 in enumerate(arrays):
            if c1>c2 : continue
            
            l, nl = so_spectra.read_ps("%s/mean_%sx%s_%s_noise.dat" % (noise_dir, ar1, ar2, survey), spectra=spectra)
            nl_t = nl["TT"][:lmax]
            nl_pol = (nl["EE"][:lmax] + nl["BB"][:lmax])/2
            l = l[:lmax]

            
            nl_array_t[c1, c2, :] = nl_t * nsplits *  2 * np.pi / (l * (l + 1))
            nl_array_pol[c1, c2, :] = nl_pol * nsplits *  2 * np.pi / (l * (l + 1))

    for i in range(lmax):
        nl_array_t[:,:,i] = fill_sym_mat(nl_array_t[:,:,i])
        nl_array_pol[:,:,i] = fill_sym_mat(nl_array_pol[:,:,i])

    return l, nl_array_t, nl_array_pol



def get_foreground_matrix(fg_dir, all_freqs, lmax):
    
    """This function uses the best fit foreground power spectra
    and generate a three dimensional array of foregroung power spectra [nfreqs, nfreqs, lmax].
    The different entries ([i,j,:]) of the array contains the fg power spectra for the different
    frequency channel pairs.
    for example fl_array_T[0,0,:] =>  fl_{f_{0},f_{0}),  fl_array_T[0,1,:] =>  fl_{f_{0},f_{1})
    this allows to have correlated fg between different frequency channels.
    (Not that for now, no fg are including in pol)
        
    Parameters
    ----------
    fg_dir : string
      the folder containing the foreground power spectra
    all_freqs: 1d array of string
      the frequencies we consider
    lmax: integer
      the maximum multipole for the noise power spectra
    """

    nfreqs = len(all_freqs)
    fl_array = np.zeros((nfreqs, nfreqs, lmax))
    
    for c1, freq1 in enumerate(all_freqs):
        for c2, freq2 in enumerate(all_freqs):
            if c1 > c2 : continue
            
            l, fl_all = np.loadtxt("%s/fg_%sx%s_TT.dat"%(fg_dir, freq1, freq2), unpack=True)
            fl_all *=  2 * np.pi / (l * (l + 1))
            
            fl_array[c1, c2, 2:lmax] = fl_all[:lmax-2]

    for i in range(lmax):
        fl_array[:,:,i] = fill_sym_mat(fl_array[:,:,i])

    return l, fl_array

def multiply_alms(alms, bl, ncomp):
    
    """This routine mutliply the alms by a function bl
        
    Parameters
    ----------
    alms : 1d array
      the alms to be multiplied
    bl : 1d array
      the function to multiply the alms
    ncomp: interger
      the number of components
      ncomp = 3 if T,Q,U
      ncomp = 1 if T only
    """
    
    alms_mult = alms.copy()
    if ncomp == 1:
        alms_mult = hp.sphtfunc.almxfl(alms_mult, bl)
    else:
        for i in range(ncomp):
            alms_mult[i] = hp.sphtfunc.almxfl(alms_mult[i], bl)
    return alms_mult


def generate_noise_alms(nl_array_t, lmax, n_splits, ncomp, nl_array_pol=None, dtype=np.complex128):
    
    """This function generates the alms corresponding to the noise power spectra matrices
    nl_array_t, nl_array_pol. The function returns a dictionnary nlms["T", i].
    The entry of the dictionnary are for example nlms["T", i] where i is the index of the split.
    note that nlms["T", i] is a (narrays, size(alm)) array, it is the harmonic transform of
    the noise realisation for the different frequencies.
    
    Parameters
    ----------
    nl_array_t : 3d array [narrays, narrays, lmax]
      noise power spectra matrix for temperature data
    
    lmax : integer
      the maximum multipole for the noise power spectra
    n_splits: integer
      the number of data splits we want to simulate
    ncomp: interger
      the number of components
      ncomp = 3 if T,Q,U
      ncomp = 1 if T only
    nl_array_pol : 3d array [narrays, narrays, lmax]
      noise power spectra matrix for polarisation data
      (in use if ncomp==3)
    """
    
    nlms = {}
    if ncomp == 1:
        for k in range(n_splits):
            nlms[k] = curvedsky.rand_alm(nl_array_t,lmax=lmax, dtype=dtype)
    else:
        for k in range(n_splits):
            nlms["T", k] = curvedsky.rand_alm(nl_array_t, lmax=lmax, dtype=dtype)
            nlms["E", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax, dtype=dtype)
            nlms["B", k] = curvedsky.rand_alm(nl_array_pol, lmax=lmax, dtype=dtype)
    
    return nlms

def remove_mean(so_map, window, ncomp):
    
    """This function removes the mean value of the map after having applied the
    window function
    Parameters
    ----------
    so_map : so_map
      the map we want to subtract the mean from
    window : so_map or so_map tuple
      the window function, if ncomp=3 expect
      (win_t,win_pol)
    ncomp : integer
      the number of components
      ncomp = 3 if T,Q,U
      ncomp = 1 if T only
      
     """
    
    if ncomp == 1:
        so_map.data -= np.mean(so_map.data * window.data)
    else:
        so_map.data[0] -= np.mean(so_map.data[0] * window[0].data)
        so_map.data[1] -= np.mean(so_map.data[1] * window[1].data)
        so_map.data[2] -= np.mean(so_map.data[2] * window[1].data)

    return so_map

def deconvolve_tf(lb, ps, tf1, tf2, ncomp, lmax=None):

    """This function deconvolves the transfer function
    Parameters
    ----------
    ps : dict or 1d array
      the power spectra with tf applied
    tf1 : 1d array
      transfer function of map1
    tf2 : 1d array
      transfer function of map2
    ncomp : integer
        the number of components
        ncomp = 3 if T,Q,U
        ncomp = 1 if T only

    """
    tf = tf1 * tf2
     
    if lmax is not None:
        id = np.where(lb < lmax)
        tf = tf[id]
        
    if ncomp == 1:
        ps /= tf
    else:
        spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
        for spec in spectra:
            ps[spec] /= tf
            
    return ps


def is_symmetric(mat, tol=1e-8):
    return np.all(np.abs(mat-mat.T) < tol)

def is_pos_def(mat):
    return np.all(np.linalg.eigvals(mat) > 0)
    
    
    
def fast_cov_coupling(sq_win_alms_dir,
                      na_r,
                      nb_r,
                      nc_r,
                      nd_r,
                      lmax,
                      l_exact=None,
                      l_band=None,
                      l_toep=None):

    if l_toep is None: l_toep = lmax
    if l_band is None: l_band = lmax
    if l_exact is None: l_exact = lmax
    
    try:
        alm_TaTc = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, na_r, nc_r))
    except:
        alm_TaTc = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, nc_r, na_r))

    try:
        alm_TbTd = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, nb_r, nd_r))
    except:
        alm_TbTd = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, nd_r, nb_r))

    try:
        alm_TaTd = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, na_r, nd_r))
    except:
        alm_TaTd = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, nd_r, na_r))

    try:
        alm_TbTc = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, nb_r, nc_r))
    except:
        alm_TbTc = np.load("%s/alms_%sx%s.npy"  % (sq_win_alms_dir, nc_r, nb_r))

    wcl = {}
    wcl["TaTcTbTd"] = hp.alm2cl(alm_TaTc, alm_TbTd)
    wcl["TaTdTbTc"] = hp.alm2cl(alm_TaTd, alm_TbTc)
    
    l = np.arange(len(wcl["TaTcTbTd"]))
    wcl["TaTcTbTd"] *= (2 * l + 1) / (4 * np.pi)
    wcl["TaTdTbTc"] *= (2 * l + 1) / (4 * np.pi)

    coupling = np.zeros((2, lmax, lmax))
     
    cov_fortran.calc_cov_spin0(wcl["TaTcTbTd"], wcl["TaTdTbTc"], l_exact, l_band, l_toep,  coupling.T)
     
    coupling_dict = {}

    for id_cov, name in enumerate(["TaTcTbTd", "TaTdTbTc"]):
        if l_toep < lmax:
            coupling[id_cov] = so_mcm.format_toepliz_fortran(coupling[id_cov], l_toep, lmax)
        mcm_fortran.fill_upper(coupling[id_cov].T)
        coupling_dict[name] = coupling[id_cov]

    list1 = ["TaTcTbTd", "PaPcPbPd", "TaTcPbPd", "PaPcTbTd",
             "TaPcTbPd", "TaTcTbPd", "TaPcTbTd", "TaPcPbTd",
             "TaPcPbPd", "PaPcTbPd", "TaTcPbTd", "PaTcTbTd",
             "PaTcPbTd", "PaTcTbPd", "PaTcPbPd", "PaPcPbTd"]
            
    list2 = ["TaTdTbTc", "PaPdPbPc", "TaPdPbTc", "PaTdTbPc",
             "TaPdTbPc", "TaPdTbTc", "TaTdTbPc", "TaTdPbPc",
             "TaPdPbPc", "PaPdTbPc", "TaTdPbTc", "PaTdTbTc",
             "PaTdPbTc", "PaPdTbTc", "PaPdPbTc", "PaTdPbPc"]
     
    for id1 in list1:
        coupling_dict[id1] = coupling_dict["TaTcTbTd"]
    for id2 in list2:
        coupling_dict[id2] = coupling_dict["TaTdTbTc"]
            
    return coupling_dict


def covariance_element(coupling, id_element, ns, ps_all, nl_all, binning_file, mbb_inv_ab, mbb_inv_cd):
    """
    This routine deserves some explanation
    We want to compute the covariance between two power spectra
    C1 = Wa * Xb, C2 =  Yc * Zd
    Here W, X, Y, Z can be either T or E and a,b,c,d will be an index
    corresponding to the survey and array we consider so for example a = s17_pa5_150 or a = dr6_pa4_090
    The formula for the analytic covariance of C1, C2 is given by
    Cov( Wa * Xb,  Yc * Zd) = < Wa Yc> <Xb Zd>  + < Wa Zd> <Xb Yc> (this is just from the wick theorem)
    In practice we need to include the effect of the mask (so we have to introduce the coupling dict D)
    and we need to take into account that we use only the cross power spectra, that is why we use the chi function
    Cov( Wa * Xb,  Yc * Zd) = D(Wa*Yc,Xb Zd) chi(Wa,Yc,Xb Zd) +  D(Wa*Zd,Xb*Yc) chi(Wa,Zd,Xb,Yc)
    
    Parameters
    ----------
    coupling : dictionnary
      a dictionnary that countains the coupling terms arising from the window functions
    id_element : list
      a list of the form [a,b,c,d] where a = dr6_pa4_090, etc, this identify which pair of power spectrum we want the covariance of
    ns: dict
      this dictionnary contains the number of split we consider for each of the survey
    ps_all: dict
      this dict contains the theoretical best power spectra, convolve with the beam for example
      ps["dr6&pa5_150", "dr6&pa4_150", "TT"] = bl_dr6_pa5_150 * bl_dr6_pa4_150 * (Dl^{CMB}_TT + fg_TT)
    nl_all: dict
      this dict contains the estimated noise power spectra, note that it correspond to the noise power spectrum per split
      e.g nl["dr6&pa5_150", "dr6&pa4_150", "TT"]
    binning_file:
      a binning file with three columns bin low, bin high, bin mean
    mbb_inv_ab and mbb_inv_cd:
      the inverse mode coupling matrices corresponding to the C1 = Wa * Xb and C2 =  Yc * Zd power spectra
    """

    na, nb, nc, nd = id_element

    lmax = coupling["TaTcTbTd"].shape[0]
    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)
    
    speclist = ["TT", "TE", "ET", "EE"]
    nspec = len(speclist)
    analytic_cov = np.zeros((nspec * nbins, nspec * nbins))
    for i, (W, X) in enumerate(speclist):
        for j, (Y, Z) in enumerate(speclist):
        
            id0 = W + "a" + Y + "c"
            id1 = X + "b" + Z + "d"
            id2 = W + "a" + Z + "d"
            id3 = X + "b" + Y + "c"
            
            M = coupling[id0.replace("E","P") + id1.replace("E","P")] * chi(na, nc, nb, nd, ns, ps_all, nl_all, W + Y + X + Z)
            M += coupling[id2.replace("E","P") + id3.replace("E","P")] * chi(na, nd, nb, nc, ns, ps_all, nl_all, W + Z + X + Y)
            analytic_cov[i * nbins: (i + 1) * nbins, j * nbins: (j + 1) * nbins] = so_cov.bin_mat(M, binning_file, lmax)

    mbb_inv_ab = so_cov.extract_TTTEEE_mbb(mbb_inv_ab)
    mbb_inv_cd = so_cov.extract_TTTEEE_mbb(mbb_inv_cd)

    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)
    
    return analytic_cov

def covariance_element_beam(id_element, ps_all, norm_beam_cov, binning_file, lmax):
    """
    This routine compute the contribution from beam errors to the analytical covariance of the power spectra
    We want to compute the beam covariance between the two spectra
    C1 = Wa * Xb, C2 =  Yc * Zd
    Here W, X, Y, Z can be either T or E and a,b,c,d will be an index
    corresponding to the survey and array we consider so for example a = dr6_pa5_150 or a = dr6_pa4_090
    The formula for the analytic covariance of C1, C2 is given by
    let's denote the normalised beam covariance <BB>_ac = < delta B_a delta B_c >/np.outer(B_a, B_c)
    
    Cov( Wa * Xb,  Yc * Zd) = Dl^{WaXb} Dl^{YcZd} ( <BB>_ac + <BB>_ad + <BB>_bc + <BB>_bd )
       
       Parameters
       ----------
    id_element : list
        a list of the form [a,b,c,d] where a = dr6_pa4_090, etc, this identify which pair of power spectrum we want the covariance of
    ps_all: dict
        this dict contains the theoretical best power spectra, convolve with the beam for example
        ps["dr6&pa5_150", "dr6&pa4_150", "TT"] = bl_dr6_pa5_150 * bl_dr6_pa4_150 * (Dl^{CMB}_TT + fg_TT)
    norm_beam_cov: dict
        this dict contains the normalized beam covariance for each survey and array
    binning_file: str
        a binning file with three columns bin low, bin high, bin mean
    lmax: int
        the maximum multipole to consider
    """
    na, nb, nc, nd = id_element

    sv_alpha, ar_alpha = na.split("&")
    sv_beta, ar_beta = nb.split("&")
    sv_gamma, ar_gamma = nc.split("&")
    sv_eta, ar_eta = nd.split("&")

    bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)

    speclist = ["TT", "TE", "ET", "EE"]
    
    nspec = len(speclist)
    analytic_cov_from_beam = np.zeros((nspec * nbins, nspec * nbins))
    for i, spec1 in enumerate(speclist):
        for j, spec2 in enumerate(speclist):
    
            M =  (delta2(na, nc) + delta2(na, nd)) * norm_beam_cov[sv_alpha, ar_alpha]
            M += (delta2(nb, nc) + delta2(nb, nd)) * norm_beam_cov[sv_beta, ar_beta]
            M *=  np.outer(ps_all[na, nb, spec1], ps_all[nc, nd, spec2])
            
            analytic_cov_from_beam[i * nbins: (i + 1) * nbins, j * nbins: (j + 1) * nbins] = so_cov.bin_mat(M, binning_file, lmax)

    return analytic_cov_from_beam
    
    
def chi(alpha, gamma, beta, eta, ns, Dl, DNl, id="TTTT"):
    """doc not ready yet
    """
    
    sv_alpha, ar_alpha = alpha.split("&")
    sv_beta, ar_beta = beta.split("&")
    sv_gamma, ar_gamma = gamma.split("&")
    sv_eta, ar_eta = eta.split("&")
    
    AB = id[0] + id[1]
    CD = id[2] + id[3]
    chi = Dl[alpha, gamma, AB] * Dl[beta, eta, CD]
    chi += Dl[alpha, gamma, AB] * DNl[beta, eta, CD] * f(sv_beta, sv_eta, sv_alpha, sv_gamma, ns)
    chi += Dl[beta, eta, CD] * DNl[alpha, gamma, AB] * f(sv_alpha, sv_gamma, sv_beta, sv_eta, ns)
    chi += g(sv_alpha, sv_gamma, sv_beta, sv_eta, ns) * DNl[alpha, gamma, AB] * DNl[beta, eta, CD]
    chi= symm_power(chi, mode="arithm")

    return chi

def delta2(a, b):
    """Simple delta function
    """

    if a == b:
        return 1
    else:
        return 0

def delta3(a, b, c):
    """Delta function (3 variables)
    """

    if (a == b) & (b == c):
        return 1
    else:
        return 0

def delta4(a, b, c, d):
    """Delta function (4 variables)
    """

    if (a == b) & (b == c) & (c == d):
        return 1
    else:
        return 0

def f(a, b, c, d, ns):
    """f combination factor in the covariance computation
    """

    result = 1. * ns[a] * (ns[c] * ns[d] * delta2(a, b) - ns[c] * delta3(a, b, d) - ns[d] * delta3(a, b, c) + delta4(a, b, c, d))
    result /= (ns[c] * ns[d] * (ns[a] - delta2(a, c)) * (ns[b] - delta2(b, d)))
    return result

def g(a, b, c, d, ns):
    """g combination factor in the covariance computation
    """

    result = 1. * ns[a] * (ns[c] * delta2(a,b) * delta2(c, d) - delta4(a, b, c, d))
    result /= (ns[a] * ns[b] * (ns[c] - delta2(a, c)) * (ns[d] - delta2(b, d)))
    return result


def symm_power(Clth, mode="arithm"):
    """Take a power spectrum Cl and return a symmetric array C_l1l2=f(Cl)

    Parameters
    ----------
    Clth: 1d array
      the power spectrum to be made symmetric
    mode : string
      geometric or arithmetic mean
      if geo return C_l1l2 = sqrt( |Cl1 Cl2 |)
      if arithm return C_l1l2 = (Cl1 + Cl2)/2
    """

    if mode == "geo":
        return np.sqrt(np.abs(np.outer(Clth, Clth)))
    if mode == "arithm":
        return np.add.outer(Clth, Clth) / 2

def interactive_covariance_comparison(analytic_cov, mc_cov, spec_list, binning_file, lmax, cov_plot_dir, multistep_path, corr_range=0.3, log=False):

    """
    This routine compare analytic covariance matrices with the ones from montecarlo simulation
    For a typical covariance matrix of ACT dr6, it compare more than 3000 nbinsxnbins covariance matrix blocs.
    This produces the plots and write a html page: covariance.html with an interactive java script interface: multistep2.js
    you can go from one plot to the other by pressing c/v on your keyboard
    
    Parameters
    ---------
    analytic_cov: 2d array
        the analytic covariance matrix
    mc_cov:  2d array
        the covariance matrix estimated with montecarlo sim
    spec_list: list of str
        the list of the different spectra to consider
    binning_file: str
        the binning file used
    lmax: int
        the maximum multipole to consider
    cov_plot_dir: str
        the directory where we write the plot to disk
    multistep_path: str
        the path to the javascript multistep2.js that render the html interactive
    corr_range: float
        should be between -1, 1 the max of the colorscale for the correlation matrix plot
    log: boolean
        wether to plot the diag of the cov with a log scale
    """

    pspy_utils.create_directory(cov_plot_dir)

    bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)

    analytic_corr = so_cov.cov2corr(analytic_cov)
    mc_corr = so_cov.cov2corr(mc_cov)

    os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, cov_plot_dir))
    file = "%s/covariance.html" % (cov_plot_dir)
    g = open(file, mode="w")
    g.write('<html>\n')
    g.write('<head>\n')
    g.write('<title> covariance comparison </title>\n')
    g.write('<script src="multistep2.js"></script>\n')
    g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
    g.write('<style> \n')
    g.write('body { text-align: center; } \n')
    g.write('img { width: 100%; max-width: 1200px; } \n')
    g.write('</style> \n')
    g.write('</head> \n')
    g.write('<body> \n')
    g.write('<div class=sub>\n')

    n_spec = int(analytic_cov.shape[0] / nbins)
    count = 0
    for ispec in range(n_spec):
        for jspec in range(ispec):
            sub_analytic_cov = analytic_cov[ispec * nbins: (ispec + 1) * nbins, jspec * nbins: (jspec + 1) * nbins]
            sub_mc_cov = mc_cov[ispec * nbins: (ispec + 1) * nbins, jspec * nbins: (jspec + 1) * nbins]

            sub_analytic_corr = analytic_corr[ispec * nbins: (ispec + 1) * nbins, jspec * nbins: (jspec + 1) * nbins]
            sub_mc_corr = mc_corr[ispec * nbins: (ispec + 1) * nbins, jspec * nbins: (jspec + 1) * nbins]

            str = "cov_%03d.png" % (count)

            fig = plt.figure(figsize=(16, 10), constrained_layout=True)
            spec = fig.add_gridspec(2, 2)
            ax0 = fig.add_subplot(spec[0, :])
            plt.title("cov(%s , %s)" % (spec_list[ispec], spec_list[jspec]), fontsize=22)
            if log == True:
                plt.semilogy()
            plt.plot(sub_analytic_cov.diagonal(), label = "analytic")
            plt.plot(sub_mc_cov.diagonal(), '.', label = "montecarlo")
            plt.legend()
            ax10 = fig.add_subplot(spec[1, 0])
            plt.imshow(sub_analytic_corr, vmin=-corr_range, vmax=corr_range, cmap='bwr')
            plt.xticks(np.arange(nbins)[::15],lb[::15].astype(int))
            plt.yticks(np.arange(nbins)[::10],lb[::10].astype(int))
            plt.colorbar()
            ax11 = fig.add_subplot(spec[1, 1])
            plt.imshow(sub_mc_corr, vmin=-corr_range, vmax=corr_range, cmap='bwr')
            plt.xticks(np.arange(nbins)[::15],lb[::15].astype(int))
            plt.yticks(np.arange(nbins)[::10],lb[::10].astype(int))
            plt.colorbar()
            plt.savefig("%s/%s" % (cov_plot_dir, str))
            plt.clf()
            plt.close()

            g.write('<div class=sub>\n')
            g.write('<img src="' + str + '"  /> \n')
            g.write('</div>\n')

            count += 1

    g.write('</body> \n')
    g.write('</html> \n')
    g.close()


