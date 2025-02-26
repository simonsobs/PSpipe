#!/usr/bin/env bash

# This bash script is used to run the dory src subtraction code

# Path to dory.py (part of the tenki python package)
dory_path=${TENKI_PATH}/point_sources

# Path to npipe maps & beams
map_path=planck_projected
beam_path=beams/npipe

# Path to the input point source catalog
ps_catalog_path=catalogs/cat_skn_090_20220526_nightonly_ordered.txt
# Note that we chose here to use the 90 GHz catalog

# Path to galactic mask
mask_path=/global/cfs/cdirs/act/data/tlouis/dr6v4/masks/mask_galactic_equatorial_car_gal080_apo0_reverse.fits

# dory.py parameters
tile_size=3000
pad_size=30
hack=6000
## Strength of the prior loaded from the input ps_catalog
prior=1
## Sources with SNR < sigma_cat (in the input catalog) are excluded from the fit
sigma_cat=15
## Sources with SNR < sigma_sub (in the catalog produced during the fitting step)
# are excluded from the subtraction
sigma_sub=2

# Splits and frequencies
splits=("A" "B")
freqs=("100" "143" "217" "353")

# Using 256 cpus, this should run in ~10mins per map
# and produce srcfree and model maps
for freq in ${freqs[@]}; do
  for split in ${splits[@]}; do
    map_file=${map_path}/npipe6v20${split}_f${freq}_map.fits
    ivar_file=${map_path}/npipe6v20${split}_f${freq}_ivar.fits
    beam_file=${beam_path}/bl_T_npipe_${freq}_coadd_pixwin.dat

    out_map_file=${map_path}/npipe6v20${split}_f${freq}_map_srcfree.fits
    out_map_model_file=${map_path}/npipe6v20${split}_f${freq}_map_model.fits

    out_cat_path=${map_path}/catalogs/cats_${freq}${split}

    time srun -n 64 -c 4 --cpu_bind=cores python "${dory_path}/dory.py" fit "${map_file}" "${ivar_file}" "${ps_catalog_path}" "${out_cat_path}" -R "tile:${tile_size}" -f "${freq}" -b "${beam_file}" --ncomp 3 --hack ${hack} -P ${prior} -m "${mask_path}" -s ${sigma_cat} --maxcorrlen 1000
    time srun -n 1 -c 256 --cpu_bind=cores python "${dory_path}/dory.py" subtract "${map_file}" "${out_cat_path}/cat.fits" "${out_map_file}" "${out_map_model_file}" -R "tile:${tile_size}" -f "${freq}" -b "${beam_file}" --ncomp 3 -s ${sigma_sub}
  done
done
