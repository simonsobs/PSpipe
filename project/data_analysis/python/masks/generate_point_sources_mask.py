import sys

import numpy as np
import pandas as pd
from pspy import so_dict, so_map

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

binary = so_map.read_map(d["survey"])
if binary.data.ndim > 2:
    # Only use temperature
    binary.data = binary.data[0]
binary.data[:] = 1

# Sigurd point sources
if "point_source_file" in d:
    print("Adding point sources...")
    df = pd.read_table(d["point_source_file"], escapechar="#", sep="\s+")
    high_flux_good_SNR = (df.Tflux > d.get("point_source_Tflux", 15)) & (
        df.SNR > d.get("point_source_SNR", 5)
    )
    df = df[high_flux_good_SNR]
    coordinates = np.deg2rad([df.dec, df.ra])
    mask = so_map.generate_source_mask(binary, coordinates, d.get("point_source_radius", 8))

# Monster sources
if "monster_source_file" in d:
    print("Adding monster point sources...")
    df = pd.read_csv(d["monster_source_file"], comment="#")
    for index, row in df.iterrows():
        mask.data *= so_map.generate_source_mask(
            binary, np.deg2rad([row.dec, row.ra]), row.radius
        ).data

# Dust
if "dust_file" in d:
    print("Adding dust sources...")
    dust = so_map.read_map(d["dust_file"])
    mask.data *= dust.data

print("Writing mask...")
mask.write_map(d["output_file"])
mask.downgrade(4)
mask.plot(file_name=d["output_file"].replace(".fits", ""))
