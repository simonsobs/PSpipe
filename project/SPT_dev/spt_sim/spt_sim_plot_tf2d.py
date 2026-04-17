import pylab as plt
import numpy as np
import healpy as hp
import sys
from pspy import pspy_utils, so_dict
from pspipe_utils import log

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
    
    def triangle_plot(alm, lmax, vmin, vmax, real=real, cmap="seismic", xlims=xlims, ylims=ylims, title=title, fig_file=fig_file):
    
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
        plt.savefig(fig_file)
        plt.clf()
        plt.close()

    if alms.ndim != 1:
        for i in range(len(alms)):
            triangle_plot(alms[i], lmax, vmin, vmax, title= title + "_%d" % i)
    else:
        triangle_plot(alms, lmax, vmin, vmax, title=title)


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

survey = "spt"
tf_dir = "tf2d"
arrays_spt = d["arrays_spt"]

plot_dir = "plots/tf2d"
pspy_utils.create_directory(plot_dir)

vmax = {}
vmax["TT"] = 6000
vmax["EE"] = 60
vmax["BB"] = 0.2

ps_2d_list = {}
for iii in range(d["iStart"], d["iStop"] + 1):
    for ar in arrays_spt:
        for i, comp in enumerate(["TT", "EE", "BB"]):
            for filt in [ "nofilter", "filter"]:
            
                if iii == 0: ps_2d_list[ar, comp, filt] = []
                
                ps_2d = np.load(f"{tf_dir}/tf2d_{comp}_{ar}_{filt}_{iii:05d}.npy")
                ps_2d_list[ar, comp, filt] += [ps_2d]

for ar in arrays_spt:
    for i, comp in enumerate(["TT", "EE", "BB"]):
        for filt in [ "nofilter", "filter"]:
        
            ps_2d_mean = np.mean(ps_2d_list[ar, comp, filt], axis=0)
            show_alm_triangle(ps_2d_mean, d["lmax"], 0, vmax[comp], xlims=[0,2000], ylims=[0,2000], title=f"{ar} {comp} {filt}", fig_file = f"{plot_dir}/{ar}_{comp}_{filt}.png" )
