import numpy as np
from matplotlib import pyplot as plt
from pixell import enmap, enplot, reproject
from pspy import so_map
import healpy as hp
from os.path import join as opj

catalog = np.loadtxt("maps/catalogs/cat.txt").T

save_dir = ""
spt_mask_dir = opj(save_dir, "masks")

header = ['ra', 'dec', 'SNR', 'Tamp', 'dTamp', 'Qamp', 'dQamp', 'Uamp', 'dUamp', 'Tflux', 'dTflux', 'Qflux', 'dQflux', 'Uflux', 'dUflux', 'npix', 'status']
catalog_dict = {k:v for k, v in zip(header, catalog)}

car_template = so_map.from_enmap(enmap.read_map(opj(spt_mask_dir, 'pixel_mask_binary_borders_only_CAR.fits')))

SNR_cut = 5

for flux_cut in [100, 50, 15]:
    flux_mask = (catalog_dict["SNR"] >  5) & (catalog_dict["Tflux"] > flux_cut)

    ps_mask = so_map.generate_source_mask(
        car_template, 
        coordinates=np.deg2rad([np.array(catalog_dict['dec'][flux_mask]), np.array(catalog_dict['ra'][flux_mask])]), 
        point_source_radius_arcmin=12
    ).data
    enmap.write_map(opj(spt_mask_dir, f"point_source_test_{flux_cut}_CAR.fits"), ps_mask)
    plot = enplot.get_plots(
        ps_mask, ticks=10, mask=0, colorbar=True, downgrade=2, range=(1)
    )
    enplot.write(opj(spt_mask_dir, f"point_source_test_{flux_cut}_CAR"), plot)
    
    ps_mask_healpix = reproject.map2healpix(ps_mask, nside=8096, method="spline")
    hp.mollview(ps_mask_healpix)
    plt.savefig(opj(spt_mask_dir, f"point_source_test_{flux_cut}.png"))
    hp.write_map(opj(spt_mask_dir, f"point_source_test_{flux_cut}.fits"), ps_mask_healpix)


