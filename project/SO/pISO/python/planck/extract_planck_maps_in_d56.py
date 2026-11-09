description="""
Code to extract Planck maps in the deep56 patch
"""
import numpy as np
from pixell import enmap
import matplotlib.pyplot as plt
from pspy import so_dict, pspy_utils
import argparse 

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--legacy', action='store_true', # default False, type bool
                    help='If given, extract legacy maps in selected window.')
parser.add_argument('--npipe', action='store_true', # default False, type bool
                    help='If given, extract npipe maps in selected window.')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

npipe = args.npipe
legacy = args.legacy
release_list = []
if npipe:
    release_list.append("npipe")
if legacy:
    release_list.append("legacy")

window_dir = d["window_dir"]
planck_freqs = ["100", "143", "217", "353"]
# all windows are the same for all channels for now
window = enmap.read_map(d["window_T_legacy_f100"])

for r in release_list:
    data_dir = d[f"maps_dir_{r}"]
    out_dir = d[f'maps_dir_{r}_extracted']
    pspy_utils.create_directory(out_dir)

    if r == "npipe":
        for f in [100, 217, 143, 353]:
            for i in ["A", "B"]:
                for c in ["map", "ivar", "map_srcfree"]:
                    pl_map = enmap.read_map(data_dir + f"npipe6v20{i}_f{f}_{c}.fits")
                    pl_extracted = enmap.extract(pl_map, window.shape, window.wcs)
                    #enplot.pshow(pl_extracted[0], downgrade = 8)
                    enmap.write_map(out_dir + f"npipe6v20{i}_f{f}_deep56_patch_{c}.fits",pl_extracted)

    if r == "legacy":
        for f in [100, 217, 143, 353]:
            for i in ["1", "2"]:
                for c in ["map", "ivar", "map_srcfree"]:
                    pl_map = enmap.read_map(data_dir + f"HFI_SkyMap_2048_R3.01_halfmission-{i}_f{f}_{c}.fits")
                    pl_extracted = enmap.extract(pl_map, window.shape, window.wcs)
                    #enplot.pshow(pl_extracted[0], downgrade = 8)
                    enmap.write_map(out_dir + f"HFI_SkyMap_2048_R3.01_halfmission-{i}_f{f}_deep56_patch_{c}.fits",pl_extracted)
