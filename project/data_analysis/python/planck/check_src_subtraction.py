"""
Iterates over the sources in the MF ACT catalog
and plot a submap centered on individual sources
of the residual map (i.e. the source subtracted map)
"""
from pspy import so_map, pspy_utils, so_map
import sys
import pandas as pd
from pspipe_utils import log

log=log.get_logger()

out_dir = "source_sub_check"
pspy_utils.create_directory(out_dir)

# Read catalog
cat_file = "/global/cfs/cdirs/act/data/tlouis/s17s18s19/catalogs/cat_skn_multifreq_20220526_nightonly.txt"
cat = pd.read_table(cat_file, escapechar="#", sep="\s+")
cat = cat.shift(1, axis=1)

# Sort by flux at 150 GHz
cat.sort_values(by="flux_T2", inplace=True, ascending=False)

# Cuts
snr_cut = 5
flux_cut = 200 #mJy

mask = (cat.snr_T >= snr_cut) & (cat.flux_T2 >= flux_cut)
cat = cat[mask]

n = len(cat)
log.info(f"Display {n} sources with 150 GHz flux >= {flux_cut:.1f} mJy and SNR >= {snr_cut:.1f}")

map_freq=100
map_split="A"
m = so_map.read_map(f"npipe_projected/npipe6v20{map_split}_f{map_freq}_map_srcfree.fits")
for i, (ra, dec) in enumerate(zip(cat.ra, cat.dec)):

    log.info(f"Doing coord {i+1} of {n}")

    ra0, ra1 = ra - 1., ra + 1.
    dec0, dec1 = dec - 1., dec + 1.

    box = so_map.get_box(ra0, ra1, dec0, dec1)
    sub = so_map.get_submap_car(m, box)

    sub.plot(file_name=f"{out_dir}/{map_freq}{map_split}_map_{i}", color_range=[250, 100, 100],
             ticks_spacing_car=0.3)
