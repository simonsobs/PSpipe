"""
Iterates over the sources in the MF ACT catalog
and plot a submap centered on individual sources
of the residual map (i.e. the source subtracted map)
"""
from pspy import so_map, pspy_utils, so_map
import sys
import pandas as pd
from pspipe_utils import log
from pspy import so_mpi

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
flux_cut = 150 #mJy
n_src_max = 100

mask = (cat.snr_T >= snr_cut) & (cat.flux_T2 >= flux_cut)
cat = cat[mask]

ra_list = [ra for i, ra in enumerate(cat.ra) if i < n_src_max]
dec_list = [dec for i, dec in enumerate(cat.dec) if i < n_src_max]

n = len(ra_list)
log.info(f"Display {n} sources with 150 GHz flux >= {flux_cut:.1f} mJy and SNR >= {snr_cut:.1f}")

release = "npipe"
map_dir = "planck_projected"
if release == "legacy":
    map_root = "HFI_SkyMap_2048_R3.01_halfmission-{}_f{}_map.fits"
    map_splits = [1, 2]

if release == "npipe":
    map_root = "npipe6v20{}_f{}_map.fits"
    map_splits = ['A', 'B']

map_freqs = [100, 143, 217]

for freq in map_freqs:
    for split in map_splits:

        map_file = map_root.format(split, freq)
        map_path = f"{map_dir}/{map_file}"

        m = so_map.read_map(map_path)
        m_srcfree = so_map.read_map(map_path.replace("map.fits", "map_srcfree.fits"))

        so_mpi.init(True)
        subtasks = so_mpi.taskrange(imin=0, imax=n-1)

        for task in subtasks:

            log.info(f"Task {task} of {n-1}")

            ra, dec = ra_list[task], dec_list[task]

            ra0, ra1 = ra - 1., ra + 1.
            dec0, dec1 = dec - 1., dec + 1.

            box = so_map.get_box(ra0, ra1, dec0, dec1)
            sub = so_map.get_submap_car(m, box)
            sub_srcfree = so_map.get_submap_car(m_srcfree, box)

            sub.plot(file_name=f"{out_dir}/{release}_f{freq}_{split}_{task:03d}", 
                     color_range=[250, 100, 100],
                     ticks_spacing_car=0.3
            )
            sub_srcfree.plot(file_name=f"{out_dir}/{release}_srcfree_f{freq}_{split}_{task:03d}", 
                     color_range=[250, 100, 100],
                     ticks_spacing_car=0.3
            )