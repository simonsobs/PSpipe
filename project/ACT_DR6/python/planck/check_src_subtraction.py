"""
Iterates over the sources in the MF ACT catalog
and plot a submap centered on individual sources
of the residual map (i.e. the source subtracted map)
"""
from pspy import so_map, pspy_utils, so_map, so_mpi, so_dict
import numpy as np
import pandas as pd
import os, sys
import pspipe
from pspipe_utils import log


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

map_dir = "planck_projected"

planck_version = d["planck_version"]

out_dir = "plots/source_sub_check"
pspy_utils.create_directory(out_dir)

# Read catalog
cat_file = d["source_catalog"]
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

if planck_version == "legacy":
    map_root = "HFI_SkyMap_2048_R3.01_halfmission-{}_f{}_map.fits"
    map_splits = [1, 2]

if planck_version == "npipe":
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

            sub.plot(file_name=f"{out_dir}/{planck_version}_f{freq}_{split}_{task:03d}",
                     color_range=[250, 100, 100],
                     ticks_spacing_car=0.6
            )
            sub_srcfree.plot(file_name=f"{out_dir}/{planck_version}_srcfree_f{freq}_{split}_{task:03d}",
                     color_range=[250, 100, 100],
                     ticks_spacing_car=0.6
            )


# Generate HTML file for visualization
ids_src = np.arange(n)
multistep_path = os.path.join(os.path.dirname(pspipe.__file__), "js")
os.system(f"cp {multistep_path}/multistep2.js {out_dir}/")

maps = [f"{freq}_{split}" for freq in map_freqs for split in map_splits]

filename = f"{out_dir}/{planck_version}_sources.html"
g = open(filename, mode='w')
g.write('<html>\n')
g.write('<head>\n')
g.write(f'<title> Source subtraction for Planck {planck_version} </title>\n')
g.write(f'<script src="multistep2.js"></script>\n')
g.write('<script> add_step("src_id", ["c","v"]) </script> \n')
g.write('<script> add_step("srcfree", ["j","k"]) </script> \n')
g.write('<script> add_step("map", ["a","z"]) </script> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write(f'<h1> Source subtraction for Planck {planck_version} </h1>')
g.write('<p> This webpage display the 100 brightest sources at 150 GHz, with a flux higher than 150mJy detected with a SNR > 5. </p> \n')
g.write('<p> You can switch between sources (c/v), between the map and source free map (j/k) and between the maps (a/z). </p> \n')
g.write('<div class=src_id> \n')
for src_id in ids_src:

    g.write('<div class=srcfree>\n')
    for type in ["", "_srcfree"]:

        prefix = f"{planck_version}{type}"
        g.write('<div class=map>\n')
        for map_name in maps:

            freq, split = map_name.split("_")
            file_name = f"{prefix}_f{freq}_{split}_{src_id:03d}_T.png"
            g.write(f'<h2> {freq} GHz split {split} - {prefix} [Source no {src_id:03d}] </p> \n')
            g.write('<img src="' + file_name + '" width="40%" /> \n')
        g.write('</div>\n')

    g.write('</div>\n')
g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()
