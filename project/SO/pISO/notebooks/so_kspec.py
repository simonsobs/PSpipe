"""
Defines a "kspec" class for making 2-D angular power spectra of CMB maps (CAR pixellization)
Doc may look like its written by chatGPT but I wrote everything 🤓
"""

from pixell import enmap, enplot
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import pickle
import itertools


def test_same_geometry(enmap_1, enmap_2):
    return (enmap_1.shape == enmap_2.shape) # TODO : also check wcs

def type2fac(type: str, ell):
    """From a str in Dl, lCl or Cl, returns fac defined as :
    type = Dl / fac
    

    Args:
        type (str): Dl, lCl or Cl
        ell (_type_): list of ell, map of ell...

    Raises:
        ValueError: _description_

    Returns:
        _type_: 1 is type==Dl, else same type as ell
    """    
    if type=='Dl':
        return 1.
    elif type=='lCl':
        return np.maximum(ell, 1)
    elif type=='Cl':
        return np.maximum(ell**2, 1)
    else:
        raise ValueError('type must be Dl, lCl or Cl')

TQU_indices = {
    'T': 0,
    'Q': 1,
    'U': 2,
    'E': 0,
    'B': 1,
}

class So_Kspec:
    # self.ncomp = None
    shape = None
    wcs = None
    ncomp=None
    Nsplits = None
    lx = None
    ly = None
    lxmap = None
    lymap = None
    llims = None
    thetamap = None
    modlmap = None
    lwcs = None
    kmaps = None
    pow = None
    pow_autos = None
    pow_crosses = None
    pow_auto = None
    pow_noise = None
    
    def __init__(self):
        pass

    def copy(self):
        return deepcopy(self)
    
    def compute_EB(self):
        # Create E and B kmaps from Q and U
        self.kmaps_EB = [enmap.zeros((2, *self.shape), kmap.wcs, dtype=np.complex128) for kmap in self.kmaps]
        for i in range(len(self.kmaps_EB)):
            self.kmaps_EB[i][0] = self.kmaps[i][1] * np.cos(2 * np.deg2rad(self.thetamap)) + self.kmaps[i][2] * np.sin(2 * np.deg2rad(self.thetamap))
            self.kmaps_EB[i][1] = self.kmaps[i][1] * np.sin(2 * np.deg2rad(self.thetamap)) - self.kmaps[i][2] * np.cos(2 * np.deg2rad(self.thetamap))

        # combine kmaps for kspecs cross and autos
        self.pow_EB_crosses = []
        self.pow_EB_autos = []
        self.pow_EB = enmap.zeros(shape=(2, *self.shape), wcs=self.lwcs)
        self.pow_EB_auto = enmap.zeros(shape=(2, *self.shape), wcs=self.lwcs)
        for i1, i2 in itertools.combinations_with_replacement(range(self.Nsplits), r=2):

            pow_iter = (self.kmaps_EB[i1] * np.conj(self.kmaps_EB[i2])).real * self.modlmap**2

            if i1 != i2:
                self.pow_EB_crosses.append(pow_iter)
                self.pow_EB += pow_iter
            elif i1 == i2:
                self.pow_EB_autos.append(pow_iter)
                self.pow_EB_auto += pow_iter

        # Divide by the number of iterations since we add all iterations
        self.pow_EB /= self.Nsplits * (self.Nsplits - 1) / 2
        self.pow_EB_auto /= self.Nsplits
        self.pow_EB_noise = (self.pow_EB_auto - self.pow_EB) / self.Nsplits
        
    def axplot(self, map_to_plot=None, TQU=None, colorbar=False, type='Dl', log=False, zoom=1000, ax_to_plot=None, downgrade=2, **args):
        """Plots self.pow by default.
        Also return the imshow() mappable (for colorbar etc.)
        """
        map_to_plot = self.pow if map_to_plot is None else map_to_plot
        if map_to_plot.shape[0] == 3:
            TQU = TQU or 'T'
            if TQU in 'TQU':
                TQU_i = TQU_indices[TQU]
                map_to_plot = map_to_plot[TQU_i]
            elif TQU in 'EB':
                TQU_i = TQU_indices[TQU]
                print(TQU_i)
                map_to_plot = map_to_plot[TQU_i]
        
        fac = type2fac(type=type, ell=self.modlmap)
        map_to_plot /= fac

        if downgrade != 1:
            map_to_plot = map_to_plot.downgrade(downgrade)

        if log:
            map_to_plot = np.log10(map_to_plot)
            
        plot = ax_to_plot.imshow(np.fft.fftshift(map_to_plot), extent=self.llims, **args)
        
        if colorbar: plt.colorbar(plot, location='bottom')
        
        ax_to_plot.set_xlim(-zoom, zoom)
        ax_to_plot.set_ylim(-zoom, zoom)
        ax_to_plot.set_xlabel(r'$\ell_x$')
        ax_to_plot.set_ylabel(r'$\ell_y$')
        return plot
    
    def plot(self, **args):
        """
        Plots the pow map using enplot, much slower than axplot()
        """
        enplot.pshow(enmap.enmap(np.fft.fftshift(self.pow), self.lwcs), **args)
    
    def radial_binned_map(self, bins_edges, which_map=None, TQU=None)-> enmap.ndmap:
        """Bins a given map (self.pow by default).

        Args:
            bins_edges (_type_): ell-indices for separation between bins
            which_map (_type_, optional): Must have same pixellization as self.pow. If None uses self.pow.
            TQU (_type_, optional): If ncomp=3, choose which one to use between to I, Q and U. If None uses I.

        Returns:
            _type_: _description_
        """        
        smap = self.pow if which_map is None else which_map
        if smap.shape[0] == 3:
            TQU = TQU or 'T'
            if TQU in 'TQU':
                TQU_i = TQU_indices[TQU]
                smap = smap[TQU_i]
            elif TQU in 'EB':
                TQU_i = TQU_indices[TQU]
                smap = smap[TQU_i]
        
        
        # Define a mask in kspace using theta map
        theta_range = [0, 180]
        theta_mask = np.where((theta_range[0] <= self.thetamap.flatten()) & (theta_range[1] > self.thetamap.flatten()))
        smap_flatten = smap.flatten()[theta_mask]
        
        # Create a map of where bins are
        bin_map = np.digitize(self.modlmap, bins_edges, right=True)
        bin_map_flatten = bin_map.flatten()[theta_mask]
        
        # Bin the map and divide by the occupation number
        rbin_map = np.bincount(bin_map_flatten, weights=smap_flatten)
        bincount = np.bincount(bin_map_flatten)
        rad_binned_map = (rbin_map / bincount)[bin_map]
        return rad_binned_map

    def radial_binned_1d_spec(self, bins_edges, which_map=None, TQU=None, theta_range=None):
        smap = self.pow if which_map is None else which_map
        if smap.shape[0] == 3:
            TQU = TQU or 'T'
            if TQU in 'TQU':
                TQU_i = TQU_indices[TQU]
                smap = smap[TQU_i]
            elif TQU in 'EB':
                TQU_i = TQU_indices[TQU]
                smap = smap[TQU_i]
        
        # Define a mask in kspace using theta map
        theta_range = theta_range or [0, 180]
        theta_mask = np.where((theta_range[0] <= self.thetamap.flatten()) & (theta_range[1] > self.thetamap.flatten()))
        smap_flatten = smap.flatten()[theta_mask]
        
        # Create a map of where bins are
        bin_map = np.digitize(self.modlmap, bins_edges, right=True)
        bin_map_flatten = bin_map.flatten()[theta_mask]
        
        # Bin the map and divide by the occupation number
        bincount = np.bincount(bin_map_flatten)
        rbin_map = np.bincount(bin_map_flatten, weights=smap_flatten)
        bins_center = (bins_edges[:-1] + bins_edges[1:]) // 2
        return bins_center, (rbin_map[1:-1] / bincount[1:-1]) # First bin is before the first bin edge and last one is after last bin edge

    def subtract_rad_profile(self, bins_edges, inplace=False):
        rad_binned_map = self.radial_binned_map(bins_edges)
        pow_subtracted = self.pow - rad_binned_map
        if not inplace:
            return pow_subtracted
        else:
            self.pow = pow_subtracted

    def write_so_kspec_pickle(self, filename:str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

def make_1d_spectra_and_save(kspec:So_Kspec, bins_edges, theta_ranges, filename):
    ls = (bins_edges[1:] + bins_edges[:-1]) / 2
    # Start with saving 1d radial power spectra and noise spectra
    ps_full = {}
    ps_full_noise = {}
    for comp in ['T', 'Q', 'U']:
        ps_full[comp] = kspec.radial_binned_1d_spec(bins_edges=bins_edges, TQU=comp)[1]
        ps_full_noise[comp] = kspec.radial_binned_1d_spec(bins_edges=bins_edges, which_map=kspec.pow_noise, TQU=comp)[1]

    # Make 1d radial power and noise spectra for given theta_ranges
    ps_thetas = {}
    ps_thetas_noise = {}
    for t, theta_range in enumerate(theta_ranges):
        range_name = f'theta_{t}'
        ps_thetas[range_name] = {}
        ps_thetas_noise[range_name] = {}
        for comp in ['T', 'Q', 'U']:
            ps_thetas[range_name][comp] = kspec.radial_binned_1d_spec(bins_edges=bins_edges, TQU=comp, theta_range=theta_range)[1]
            ps_thetas_noise[range_name][comp] = kspec.radial_binned_1d_spec(bins_edges=bins_edges, which_map=kspec.pow_noise, TQU=comp, theta_range=theta_range)[1]
    
    # Put everything in a dict and save it
    save_dict = {
        'theta_range': theta_ranges,
        'ls': ls,
        'ps_full': ps_full,
        'ps_thetas': ps_thetas,
        'ps_full_noise': ps_full_noise,
        'ps_thetas_noise': ps_thetas_noise,
    }
    
    with open(filename, "wb") as f:
            pickle.dump(save_dict, f)


def read_so_kspec_pickle(filename:str):
    with open(filename, "rb") as f:
        return pickle.load(f)

def from_enmap(enmap_: enmap.ndmap) -> So_Kspec:
    kspec = So_Kspec()

    # some geometry and misc
    kspec.shape = (enmap_.shape[-2],enmap_.shape[-1])
    kspec.ncomp = 3 if enmap_.shape[0]==3 else 1
    kspec.wcs = enmap_.wcs
    kspec.ly, kspec.lx = enmap_.laxes()
    kspec.lymap, kspec.lxmap = enmap_.lmap()
    kspec.modlmap = enmap_.modlmap().astype(np.float32)
    kspec.lwcs = enmap.lwcs(enmap_.shape, enmap_.wcs)
    kspec.llims = (min(kspec.lx), max(kspec.lx), min(kspec.ly), max(kspec.ly))
    kspec.thetamap = np.rad2deg(np.arctan2(kspec.lymap, kspec.lxmap))
    
    # compute kmaps and kspecs
    kmap = enmap.fft(enmap_)
    kspec.kmaps = [kmap]
    kspec.pow = (kmap * np.conj(kmap)).real
    kspec.pow *= kspec.modlmap**2 # D_ell because why not

    return kspec

def from_enmap_list(enmap_list: list[enmap.ndmap]) -> So_Kspec:
    N_splits = len(enmap_list)
    
    kspec = So_Kspec()
    enmap_template = enmap_list[0] # assume all map should have same geometry
    kspec.shape = (enmap_template.shape[-2], enmap_template.shape[-1])
    kspec.ncomp = 3 if enmap_template.shape[0]==3 else 1
    kspec.Nsplits = N_splits
    kspec.wcs = enmap_template.wcs
    kspec.ly, kspec.lx = enmap_template.laxes()
    kspec.lymap, kspec.lxmap = enmap_template.lmap()
    kspec.modlmap = enmap_template.modlmap().astype(np.float32)
    kspec.lwcs = enmap.lwcs(kspec.shape, kspec.wcs)
    kspec.llims = (min(kspec.lx), max(kspec.lx), min(kspec.ly), max(kspec.ly))
    kspec.thetamap = np.rad2deg(np.arctan2(kspec.lymap, kspec.lxmap))
    
    # start with kmaps
    kspec.kmaps = [enmap.fft(enmap_) for enmap_ in enmap_list]
    
    # combine kmaps for kspecs cross and autos
    kspec.pow_crosses = []
    kspec.pow_autos = []
    kspec.pow = enmap.zeros(shape=enmap_template.shape, wcs=kspec.lwcs)
    kspec.pow_auto = enmap.zeros(shape=enmap_template.shape, wcs=kspec.lwcs)
    for i1, i2 in itertools.combinations_with_replacement(range(N_splits), r=2):
        assert test_same_geometry(enmap_list[i1], enmap_list[i2]), "All maps must have same geometry"

        # if kspec.ncomp==1:
        #     pow_iter = (kspec.kmaps[i1] * np.conj(kspec.kmaps[i2])).real * kspec.modlmap**2
        # elif kspec.ncomp==3:
        #     pow_iter = enmap.zeros(shape=kspec.kmaps[0].shape, wcs=kspec.kmaps[0].wcs)
        #     for i in range(3):
        #         pow_iter[i] = (kspec.kmaps[i1][i] * np.conj(kspec.kmaps[i2][i])).real * kspec.modlmap**2
        pow_iter = (kspec.kmaps[i1] * np.conj(kspec.kmaps[i2])).real * kspec.modlmap**2

        if i1 != i2:
            kspec.pow_crosses.append(pow_iter)
            kspec.pow += pow_iter
        elif i1 == i2:
            kspec.pow_autos.append(pow_iter)
            kspec.pow_auto += pow_iter

    # Divide by the number of iterations since we add all iterations
    kspec.pow /= N_splits * (N_splits - 1) / 2
    kspec.pow_auto /= N_splits
    kspec.pow_noise = (kspec.pow_auto - kspec.pow) / N_splits
    
    return kspec