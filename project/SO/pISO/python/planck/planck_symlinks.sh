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

# passbands
ln -s ${dr6_realease_passbands} ${passbands_dir}
mv ${passbands_dir}/passbands ${passbands_dir}/planck