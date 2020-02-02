#!/bin/bash

pspipe_workspace=${PSPIPE_WORKSPACE:-/WHERE/TO/WORK/pspipe_workspace}

githubs=(
    "https://github.com/simonsobs/pixell"
    "https://github.com/simonsobs/pspy"
    "https://github.com/simonsobs/PSpipe"
    "https://github.com/LSSTDESC/NaMaster"
    "https://github.com/simonsobs/LAT_MFLike"
)

mkdir -p ${pspipe_workspace}
(
    cd ${pspipe_workspace}
    for gh in "${githubs[@]}"; do
        name=$(echo ${gh} | cut -d'/' -f5)
        if [ ! -d ./${name} ]; then
            git clone ${gh}
            continue
        fi
        (cd ./${name}; git pull)
    done
)


docker run --rm -it -v ${pspipe_workspace}:/home/pspipe/workspace -p 8888:8888 simonsobs/pspipe:latest /bin/bash
