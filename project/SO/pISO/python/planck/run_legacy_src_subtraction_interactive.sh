#!/usr/bin/env bash

# This bash script is used to run the dory src subtraction code
# on Planck legacy maps. Note that we use the Planck NPIPE beams
# because legacy beams are truncated at l=4000 and the dory source
# subtraction code does not work properly with truncated beams.

# read some info of the paramfile
PARAMFILE=$1
read dory_path map_path beam_path < <(
python - <<END
from pspy import so_dict
d = so_dict.so_dict()
d.read_from_file("$PARAMFILE")
print(d["dory_path"], d["maps_dir_planck"], d["beam_dir_planck"])
END
)

# Path to the input point source catalog
ps_catalog_path=${map_path}/cat_skn_090_20220526_nightonly_ordered.txt
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
splits=("1" "2")
freqs=("100" "143" "217" "353")
# splits=("2")
# freqs=("100")

# Using 256 cpus, this should run in ~10mins per map
# and produce srcfree and model maps
for freq in ${freqs[@]}; do
  for split in ${splits[@]}; do

    map_file=${map_path}/legacy/HFI_SkyMap_2048_R3.01_halfmission-${split}_f${freq}_map.fits
    ivar_file=${map_path}/legacy/HFI_SkyMap_2048_R3.01_halfmission-${split}_f${freq}_ivar.fits
    beam_file=${beam_path}/bl_T_extended_${freq}_coadd_pixwin.dat
    out_map_file=${map_path}/legacy/HFI_SkyMap_2048_R3.01_halfmission-${split}_f${freq}_map_srcfree.fits
    out_map_model_file=${map_path}/legacy/HFI_SkyMap_2048_R3.01_halfmission-${split}_f${freq}_map_model.fits
    echo split${split}_f${freq}
    out_cat_path=${map_path}/cats_${freq}${split}
    # echo srun -n 1 -c 256 --cpu_bind=cores python "${dory_path}/dory.py" subtract "${map_file}" "${out_cat_path}/cat.fits" "${out_map_file}" "${out_map_model_file}" -R "tile:${tile_size}" -f "${freq}" -b "${beam_file}" --ncomp 3 -s ${sigma_sub}

    OMP_NUM_THREADS=64 srun -n 4 -c 64 --cpu_bind=cores python "${dory_path}/dory.py" fit "${map_file}" "${ivar_file}" "${ps_catalog_path}" "${out_cat_path}" -R "tile:${tile_size}" -f "${freq}" -b "${beam_file}" --ncomp 3 --hack ${hack} -P ${prior} -m "${mask_path}" -s ${sigma_cat} --maxcorrlen 1000
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu_bind=cores python "${dory_path}/dory.py" subtract "${map_file}" "${out_cat_path}/cat.fits" "${out_map_file}" "${out_map_model_file}" -R "tile:${tile_size}" -f "${freq}" -b "${beam_file}" --ncomp 3 -s ${sigma_sub}
  done
done
