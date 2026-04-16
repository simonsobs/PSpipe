"""
Generate SPT source mask from dory catalogs, we mask the brightest sources at the three different frequencies
Note that dory-catalog contains copies at the location of the brightest source
We also mask cluster because we will not model their frequency dependence in the likelihood
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from pspy import so_map, so_dict, so_window, pspy_utils
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

release_dir = d["release_dir"]
freqs = d["freqs_spt"]

n_tops = d["n_tops"]
point_source_radius_arcmin = d["point_source_radius_arcmin"]
apo_radius_degree = d["apo_radius_degree"]
apo_type = d["apo_type"]

binary_dir =  release_dir + "ancillary_products/generally_applicable/"

columns = ["ra", "dec", "SNR", "Tamp", "dTamp", "Qamp", "dQamp", "Uamp", "dUamp",
           "Tflux", "dTflux", "Qflux", "dQflux", "Uflux", "dUflux", "npix", "status"]

colors = {}
colors["095"] = "red"
colors["150"] = "green"
colors["220"] = "blue"

mask_dir = "my_masks"
pspy_utils.create_directory(mask_dir)


for n_top in n_tops:

    edge_map = so_map.read_map(binary_dir + "pixel_mask_apodized_borders_only.fits")
    binary = edge_map.copy()
    binary.data[:] = 1

    plt.figure(figsize=(18, 12))

    for freq in freqs:
        cat = f"catalogs_{freq}/cat_dedup_fit/cat.txt"
        df = pd.read_csv(cat, sep='\s+', comment='#', names=columns)
        
        df_top = df.nlargest(n_top, "Tamp")
        
        if freq != "220":
            df_top_neg = df.nsmallest(n_top, "Tamp")
            df_combined = pd.concat([df_top, df_top_neg])
        else:
            df_combined = df_top

    
        output_name = f"{mask_dir}/spt_catalog_{freq}_top{n_top}.txt"
        with open(output_name, 'w') as f:
            f.write(df_combined.to_string(index=False, justify="left"))
                        
        # add sources that pop up visually but were missed by the source detection
        xtra_ra = [coord[0] for coord in d["xtra_sources_ra_dec"]]
        xtra_dec = [coord[1] for coord in d["xtra_sources_ra_dec"]]
        
        all_dec = np.concatenate([df_combined["dec"], xtra_dec])
        all_ra = np.concatenate([df_combined["ra"], xtra_ra])
        coordinates = np.deg2rad([all_dec, -all_ra])


        freq_mask = so_map.generate_source_mask(binary, coordinates, point_source_radius_arcmin)
        binary.data *= freq_mask.data

        
        sizes = df_combined["Tamp"] ** 2
        plt.scatter(df_combined["ra"], df_combined["dec"], s=sizes, alpha=0.5, edgecolors=colors[freq], facecolors="none")
        

    plt.xlabel("Right Ascension (RA)")
    plt.ylabel("Declination (DEC)")
    plt.title("Catalog")
                
    plt.gca().invert_xaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{mask_dir}/catalog_top{n_top}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()
    
    binary = so_window.create_apodization(binary, apo_type, apo_radius_degree)
    binary.data *= edge_map.data

    binary.data[binary.data == 0] = hp.UNSEEN
    hp.fitsfunc.write_map(f"{mask_dir}/spt_source_mask_top{n_top}_apod.fits", binary.data, partial=True, dtype=np.float32, overwrite=True)
