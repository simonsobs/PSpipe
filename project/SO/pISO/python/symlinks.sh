#!/bin/bash

PARAMFILE=$1
read data_dir < <(
python - <<END
from pspy import so_dict
d = so_dict.so_dict()
d.read_from_file("$PARAMFILE")
print(d["data_dir"])
END
)

dr6_realease_passbands='/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/pspipe/for_planck/passbands'
freqs=("100" "143" "217" "353")

passbands_dir=${data_dir}/passbands/

# Planck passbands
ln -s ${dr6_realease_passbands} ${passbands_dir}
mv ${passbands_dir}/passbands ${passbands_dir}/planck

# DR6 passbands
ln -s /global/cfs/cdirs/cmb/data/act_dr6/dr6.02/passbands/processed/ ${passbands_dir}
mv ${passbands_dir}/processed ${passbands_dir}/dr6

# DR6 beams
dr6_beams_dir=${data_dir}/beams/dr6/
ln -s /global/cfs/cdirs/cmb/data/act_dr6/dr6.02/beams/main_beams/nominal/ ${dr6_beams_dir}
mv ${dr6_beams_dir}/nominal ${dr6_beams_dir}/main_beams

ln -s /global/cfs/cdirs/cmb/data/act_dr6/dr6.02/beams/leakage_beams/nominal/ ${dr6_beams_dir}
mv ${dr6_beams_dir}/nominal ${dr6_beams_dir}/leakage_beams