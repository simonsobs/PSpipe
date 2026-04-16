#!/usr/bin/env bash

# This bash script find the source in the SPT patch

# Path to dory.py (part of the tenki python package)
dory_path=${TENKI_PATH}/point_sources

# Path to spt projected maps & beams
map_path=projected_maps/spt
mask_path=projected_masks/pixel_mask_binary_borders_only_CAR_rev.fits

beam_path=/global/cfs/cdirs/sobs/users/tlouis/spt_data/ancillary_products/generally_applicable

# dory.py parameters
tile_size=3000
## Strength of the prior loaded from the input ps_catalog
prior=1
## Sources with SNR < sigma_cat (in the input catalog) are excluded from the fit
sigma_cat=10
# hack is used because some experiments lack power after a certain ell, at this ell we will
# just use the measurement at lower l (particularly important for the noise model)
hack=15000

freqs=("095" "150" "220")

for freq in ${freqs[@]}; do

    if [ "$freq" == "095" ]; then
	beam_val="1.6"
    elif [ "$freq" == "150" ]; then
	beam_val="1.1"
    elif [ "$freq" == "220" ]; then
	beam_val="1.0"
    else
	beam_val="1.1"
    fi
  
    map_file=${map_path}/full_${freq}ghz_CAR.fits
    ivar_file=${map_path}/ivar_${freq}ghz_CAR.fits
    beam_file=${beam_path}/beam_c26_cmb_bl_${freq}ghz.txt
    out_cat_path=catalogs_${freq}
    time srun -n 64 -c 4 --cpu_bind=cores python "${dory_path}/dory.py" find "${map_file}" "${ivar_file}"  "${out_cat_path}" -R "tile:${tile_size}" -f "${freq}" -b "${beam_file}" -m "${mask_path}" --verbose  --hack ${hack}
    time srun -n 1 -c 256 --cpu_bind=cores python  "${dory_path}/dedup_catalog.py" -s ${sigma_cat} -b "${beam_val}" --bscale=8 "${out_cat_path}/cat.txt" "${out_cat_path}/cat_dedup.txt" #avoid duplicate
    time srun -n 64 -c 4 --cpu_bind=cores python "${dory_path}/dory.py" fit "${map_file}" "${ivar_file}"  "${out_cat_path}/cat_dedup.txt"  "${out_cat_path}/cat_dedup_fit" -R "tile:${tile_size}" -f "${freq}" -b "${beam_file}" --ncomp 3  -P ${prior} -m "${mask_path}" -s ${sigma_cat}  --hack ${hack}
done
