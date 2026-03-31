from matplotlib import pyplot
import healpy, numpy, os, matplotlib, sys


matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.family"] = "DeJavu Serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
matplotlib.rcParams["legend.fontsize"] = 14
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["axes.grid.which"] = "both"
matplotlib.rcParams["grid.linestyle"] = "dotted"
matplotlib.rcParams["grid.linewidth"] = 0.35
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["axes.labelpad"] = 6
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["axes.titlepad"] = 10
matplotlib.rcParams["xtick.labelsize"] = 14
matplotlib.rcParams["ytick.labelsize"] = 14

def show_map_full_field(
        m, vmin, vmax, title, unit,
        cmap="gray", badcolor="white"):

    pyplot.figure(figsize=(13, 7), num=0, facecolor="white")
    healpy.azeqview(
        m,
        rot=(0, -59.5, 0),
        xsize=1300, ysize=700, reso=3.5, fig=0,
        half_sky=True, lamb=True,
        cmap=cmap, min=vmin, max=vmax,
        badcolor=badcolor,
        title=title, unit=unit)
    pyplot.show()
    pyplot.close()


def show_map_thumbnail(
        m, vmin, vmax, title, unit,
        cmap="gray", badcolor="white"):

    pyplot.figure(figsize=(8, 8), num=0, facecolor="white")
    healpy.gnomview(
        m,
        rot=(32, -51, 0),
        xsize=800, ysize=800, reso=0.2, fig=0,
        cmap=cmap, min=vmin, max=vmax,
        badcolor=badcolor,
        title=title, unit=unit)
    pyplot.show()
    pyplot.close()


def show_1d_functions_of_ell(
        ell, functions, labels, xlims, ylims, ylabel,
        yscale="linear", legend_loc="upper right", legend_ncols=1,
        vlines=[]):

    pyplot.figure(figsize=(8, 5))
    for function, label in zip(functions, labels):
        pyplot.plot(ell, function, label=label, alpha=0.8)
    for vline in vlines:
        pyplot.axvline(vline, color="black", linestyle="dotted")
    pyplot.yscale(yscale)
    pyplot.xlim(left=xlims[0], right=xlims[1])
    pyplot.ylim(bottom=ylims[0], top=ylims[1])
    if not (len(labels) == 1 and labels[0] == ""):
        pyplot.legend(loc=legend_loc, ncol=legend_ncols)
    pyplot.xlabel(r"$\ell$")
    pyplot.ylabel(ylabel)
    pyplot.show()
    pyplot.close()


def show_alm_triangle(
        alm, lmax, real=True,
        vmin=None, vmax=None, cmap="Oranges_r",
        xlims=None, ylims=None,
        title="Triangle"):

    import warnings
    warnings.filterwarnings("ignore")

    triangle = numpy.empty((lmax+1, lmax+1))
    triangle[:,:] = numpy.nan
    for l in range(lmax+1):
        for m in range(0, l+1):
            i = healpy.Alm.getidx(lmax, l, m)
            if real:
                triangle[m, l] = alm[i].real
            else:
                triangle[m, l] = alm[i]

    pyplot.figure(figsize=(7, 7))
    if vmin is None:
        vmin = numpy.min(triangle)
    if vmax is None:
        vmax = numpy.max(triangle)
    img = pyplot.imshow(
        triangle, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    if xlims is None:
        xlims = [0, triangle.shape[1]]
    if ylims is None:
        ylims = [0, triangle.shape[0]]
    pyplot.xlim(left=xlims[0], right=xlims[1])
    pyplot.ylim(bottom=ylims[0], top=ylims[1])
    pyplot.grid(False)
    cb = pyplot.colorbar(pad=0.03, shrink=0.8)
    pyplot.grid(True)
    pyplot.xlabel(r"$\ell$")
    pyplot.ylabel(r"$m$")
    pyplot.title(title)
    pyplot.show()
    pyplot.close()
