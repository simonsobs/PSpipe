"""
Routines for generalized map2alm and alm2map (healpix and CAR).
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from pixell import curvedsky, enmap

from pspy import so_window


def map2alm(map, niter, lmax, theta_range=None, dtype=np.complex128):
    """Map2alm transform (for healpix or CAR).

    Parameters
    ----------
    map: ``so_map``
      the map from which to compute the alm
    niter: integer
      the number of iteration performed while computing the alm
      not that for CAR niter=0 should be enough
    lmax:  integer
      the maximum multipole of the transform
    theta_range: list of 2 elements
      [theta_min,theta_max] in radian.
      for healpix pixellisation all pixel outside this range will be assumed to be zero.
    """
    if map.pixel == "HEALPIX":
        if theta_range is None:
            alm = hp.sphtfunc.map2alm(map.data, lmax=lmax, iter=niter)

        else:
            nside = hp.pixelfunc.get_nside(map.data)
            alm = curvedsky.map2alm_healpix(map.data,
                                            lmax=lmax,
                                            theta_min=theta_range[0],
                                            theta_max=theta_range[1],
                                            niter=niter)

    elif map.pixel=="CAR":
        alm = curvedsky.map2alm(map.data, lmax=lmax, niter=niter)
    else:
        raise ValueError("Map is neither a CAR nor a HEALPIX")

    alm = alm.astype(dtype)
    return alm

def alm2map(alms, template):
    """alm2map transform (for healpix and CAR).

    Parameters
    ----------
    alms: array
      a set of alms, the shape of alms should correspond to so_map.ncomp
    template: ``so_map``
      the map template
    """
    map_from_alm = template.copy()
    if map_from_alm.ncomp == 1:
        spin = 0
    else:
        spin = [0, 2]
    if map_from_alm.pixel == "HEALPIX":
        map_from_alm.data = curvedsky.alm2map_healpix(alms, map_from_alm.data, spin=spin)
    elif map_from_alm.pixel == "CAR":
        map_from_alm.data = curvedsky.alm2map(alms, map_from_alm.data, spin=spin)
    else:
        raise ValueError("Map is neither a CAR nor a HEALPIX")
    return map_from_alm

def get_alms(so_map, window, niter, lmax, theta_range=None, dtype=np.complex128, alm_conv="HEALPIX"):
    """Get a map, multiply by a window and return alms
    This is basically map2alm but with application of the
    window functions.

    Parameters
    ----------

    so_map: ``so_map``
      the data we wants alms from
    window: so_map or tuple of so_map
      a so map with the window function, if the so map has 3 components
      (for spin0 and 2 fields) expect a tuple (window,window_pol)
    theta range: list of 2 elements
      for healpix pixellisation you can specify
      a range [theta_min,theta_max] in radian. All pixel outside this range
      will be assumed to be zero.
    alm_conv: str
        default is HEALPIX, if IAU multiply U by -1
    """
    windowed_map = so_map.copy()
    if so_map.ncomp == 3:
        windowed_map.data[0] = so_map.data[0]*window[0].data
        windowed_map.data[1] = so_map.data[1]*window[1].data
        windowed_map.data[2] = so_map.data[2]*window[1].data
        if alm_conv == "IAU":
            windowed_map.data[2] = -windowed_map.data[2]
        
    if so_map.ncomp == 1:
        windowed_map.data = so_map.data * window.data
        

    alms = map2alm(windowed_map, niter, lmax, theta_range=theta_range, dtype=dtype)
    return alms


def get_pure_alms(so_map, window, spinned_windows, niter, lmax, theta_range=None, alm_conv="HEALPIX"):

    """Compute pure alms from maps and window function

    Parameters
    ----------

    so_map: ``so_map``
      the data we wants alms from
    window: so_map or tuple of so_map
      a so map with the window function, if the so map has 3 components
      (for spin0 and 2 fields) expect a tuple (window,window_pol)
    niter: integer
      the number of iteration performed while computing the alm
      not that for CAR niter=0 should be enough
    lmax:  integer
      the maximum multipole of the transform
    theta range: list of 2 elements
      for healpix pixellisation you can specify
      a range [theta_min,theta_max] in radian. All pixel outside this range
      will be assumed to be zero.
    alm_conv: str
        default is HEALPIX, if IAU multiply U by -1

    """
    

    
    T,Q,U = so_map.data[0], so_map.data[1], so_map.data[2]
    if alm_conv == "IAU":
        U = -U

    w1_plus, w1_minus, w2_plus, w2_minus = spinned_windows
    p2 = np.array([window[1].data * Q, window[1].data * U])
    p1 = np.array([(w1_plus.data * Q + w1_minus.data * U), (w1_plus.data * U - w1_minus.data * Q)])
    p0 = np.array([(w2_plus.data * Q + w2_minus.data * U), (w2_plus.data * U - w2_minus.data * Q)])

    if so_map.pixel == "CAR":
        p0 = enmap.samewcs(p0, so_map.data)
        p1 = enmap.samewcs(p1, so_map.data)
        p2 = enmap.samewcs(p2, so_map.data)

        alm = curvedsky.map2alm(T * window[0].data, lmax=lmax)
        s2eblm = curvedsky.map2alm(p2, spin=2, lmax=lmax)
        s1eblm = curvedsky.map2alm(p1, spin=1, lmax=lmax)
        s0eblm = s1eblm.copy()
        s0eblm[0] = curvedsky.map2alm(p0[0], spin=0, lmax=lmax)
        s0eblm[1] = curvedsky.map2alm(p0[1], spin=0, lmax=lmax)

    if so_map.pixel == "HEALPIX":
    
        theta_min, theta_max = theta_range if theta_range is not None else (None, None)
        
        alm  = curvedsky.map2alm_healpix(T * window[0].data, spin=0, lmax=lmax, niter=niter, theta_min=theta_min, theta_max=theta_max)
        s2eblm = curvedsky.map2alm_healpix(p2, spin=2, lmax=lmax, niter=niter, theta_min=theta_min, theta_max=theta_max)
        s1eblm = curvedsky.map2alm_healpix(p1, spin=1, lmax=lmax, niter=niter, theta_min=theta_min, theta_max=theta_max)

        s0eblm= s1eblm.copy()
        s0eblm[0] = curvedsky.map2alm_healpix(p0[0], spin=0, lmax=lmax, niter=niter, theta_min=theta_min, theta_max=theta_max)
        s0eblm[1] = curvedsky.map2alm_healpix(p0[1], spin=0, lmax=lmax, niter=niter, theta_min=theta_min, theta_max=theta_max)

    ell = np.arange(lmax+1)
    filter_1 = np.zeros(lmax+1)
    filter_2 = np.zeros(lmax+1)

    filter_1[2:] = 2 * np.sqrt(1.0 / ((ell[2:] + 2.) * (ell[2:] - 1.)))
    filter_2[2:] = np.sqrt(1.0  / ((ell[2:] + 2.) * (ell[2:] + 1.) * ell[2:] * (ell[2:] - 1.)))

    for k in range(2):
        s1eblm[k] = hp.almxfl(s1eblm[k],filter_1)
        s0eblm[k] = hp.almxfl(s0eblm[k],filter_2)

    elm_p = s2eblm[0] + s1eblm[0] + s0eblm[0]
    blm_b = s2eblm[1] + s1eblm[1] + s0eblm[1]

    return np.array([alm,elm_p,blm_b])

def show_alm_triangle(
        alms, lmax, vmin, vmax, real=True, cmap="seismic",
        xlims=None, ylims=None,
        title="Triangle", fig_file=None):
        
    """
    This routine is from the spt3g data release
    https://pole.uchicago.edu/public/data/quan26/index.html
    Parameters
    ----------
    alms: array
      a set of alms, the shape of alms should correspond to so_map.ncomp
    lmax: int
      the maximum multipole of the transform
    real: boolean
      withtout to take the real part or not
    vmin, vmax: float
      range of the colorbar
    cmap: string
      name of the colormap
    xlims, ylims: int
      x is for the \ell coordiante, y for the m coordinate
    title: str
     title, if multiple alm, will appended an integer number 0,1,2...
    """
    

    import warnings
    warnings.filterwarnings("ignore")
    
    def triangle_plot(alm, lmax, vmin, vmax, real, cmap, xlims, ylims, title, fig_file):
    
        triangle = np.empty((lmax+1, lmax+1))
        triangle[:,:] = np.nan
        for l in range(lmax+1):
            for m in range(0, l+1):
                i = hp.Alm.getidx(lmax, l, m)
                if real:
                    triangle[m, l] = alm[i].real
                else:
                    triangle[m, l] = alm[i]

        plt.figure(figsize=(7, 7))
        img = plt.imshow(
            triangle, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        if xlims is None:
            xlims = [0, triangle.shape[1]]
        if ylims is None:
            ylims = [0, triangle.shape[0]]
        plt.xlim(left=xlims[0], right=xlims[1])
        plt.ylim(bottom=ylims[0], top=ylims[1])
        plt.grid(False)
        cb = plt.colorbar(pad=0.03, shrink=0.8)
        plt.grid(True)
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$m$")
        plt.title(title)
        if fig_file is not None:
            plt.savefig(fig_file)
            plt.clf()
            plt.close()
        else:
            plt.show()
            
    if alms.ndim != 1:
        for i in range(len(alms)):
            if fig_file is not None:
                new_fig_file = fig_file + "_%d.png" % i
            else:
                new_fig_file = None
            if title is not None:
                new_title = title + "_%d" % i
            else:
                new_title = None
                
            triangle_plot(alms[i], lmax, vmin, vmax, real, cmap, xlims, ylims, new_title, new_fig_file)
    else:
        triangle_plot(alms, lmax, vmin, vmax, real, cmap, xlims, ylims, title, f"{fig_file}.png")
