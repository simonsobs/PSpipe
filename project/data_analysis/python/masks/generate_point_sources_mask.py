import sys

import numpy as np
import pandas as pd
from pspy import so_dict, so_map

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


freq = d["freq"]
binary = so_map.read_map(d["template"])
if binary.data.ndim > 2:
    # Only use temperature
    binary.data = binary.data[0]
binary.data = binary.data.astype(np.int16)
binary.data[:] = 1

flux_id = {90: 1, 150: 2, 220: 3}

f_name = "act_pts_mask"
# Sigurd point sources
if "point_source_file" in d:
    print("Adding point sources...")
    
    flux_cut, snr_cut, radius = d["point_source_Tflux"], d["point_source_SNR"], d["point_source_radius"]
    
    df = pd.read_table(d["point_source_file"], escapechar="#", sep="\s+")
    df = df.shift(1, axis = 1) # this is a hack let's find a better way
    flux_name = "flux_T%d" % flux_id[freq]
    df = df.sort_values(by=flux_name, ascending=False)[: d.get("point_source_only_keep_nth_brighest")]

    high_flux_good_SNR = (d["cal_%d" % freq] * getattr(df, flux_name) > flux_cut) & (df.snr_T > snr_cut)
  
    df = df[high_flux_good_SNR]
    coordinates = np.deg2rad([df.dec, df.ra])
    mask = so_map.generate_source_mask(binary, coordinates, radius)
    
    print("N_sources: %d" % len(df))

    f_name += "_fluxcut_%0.1fmJy_at%dGhz_rad_%0.1f" % (flux_cut, d["freq"], radius)

# Monster sources
if "monster_source_file" in d:
    print("Adding monster point sources...")
    df = pd.read_csv(d["monster_source_file"], comment="#")
    for index, row in df.iterrows():
        mask.data *= so_map.generate_source_mask(
            binary, np.deg2rad([row.dec, row.ra]), row.radius
        ).data
    f_name += "_monster"

# Dust
if "dust_file" in d:
    print("Adding dust sources...")
    dust = so_map.read_map(d["dust_file"])
    mask.data *= dust.data
    f_name += "_dust"


print("Writing mask...")
mask.write_map("%s.fits" % f_name)
mask.downgrade(4).plot(file_name=f_name)
