#!/bin/bash
#-*- mode: shell-script; -*-
#
# Copyright (C) 2020 Simons Observatory
#
# Keywords: docker, pspipe
# Requirements: docker
# Status: not intended to be distributed yet

current_dir=$(dirname $0)
pspipe_workspace=${PSPIPE_WORKSPACE:-${current_dir/\./$PWD}}
githubs=(
    "https://github.com/simonsobs/pixell"
    "https://github.com/simonsobs/pspy"
    "https://github.com/LSSTDESC/NaMaster,easier_libsharp"
    "https://github.com/simonsobs/LAT_MFLike"
    "https://github.com/simonsobs/PSpipe"
    "https://github.com/healpy/pysm"
)

function msg_error()
{
    echo -en "\\033[0;31mERROR: $@\\033[0;39m\n"
    return 0
}

function msg_notice()
{
    echo -en "\\033[0;34mNOTICE: $@\\033[0;39m\n"
    return 0
}

function msg_warning()
{
    echo -en "\\033[0;35mWARNING: $@\\033[0;39m\n"
    return 0
}

function prepare_workspace() {
    (
        cd ${pspipe_workspace}
        for gh in "${githubs[@]}"; do
            local http=$(echo ${gh} | awk -F, '{print $1}')
            local branch=$(echo ${gh} | awk -F, '{print $2}')
            local name=$(echo ${http} | awk -F/ '{print $NF}')
            if [ ! -d ${name} ]; then
                msg_notice "Cloning '${name}' into ${psipe_workspace}..."
                git clone ${http}
                [[ ! -z ${branch} ]] && (cd ${name} && git checkout origin/${branch} -b ${branch})
                continue
            else
                msg_notice "Package '${name}' already cloned."
            fi
        done
    )
    cid_file="${pspipe_workspace}/.cid"
    return 0
}

function start_docker() {
    msg_notice "Starting PSpipe docker..."
    msg_notice "Type exit when you are done"
    [[ -f ${cid_file} ]] && rm -f ${cid_file}

    local docker_options="-it -p 8888:8888"
    docker_options+=" --cidfile ${cid_file} -v ${pspipe_workspace}:/home/pspipe/workspace"

    docker run $(echo ${docker_options}) ghcr.io/simonsobs/pspipe:master /bin/bash
    return 0
}

function stop_docker() {
    msg_notice "Stopping PSpipe docker."
    if [ ! -z ${cid_file} ]; then
        cid_value=$(cat ${cid_file})
        docker commit ${cid_value} simonsobs/pspipe:master > /dev/null 2>&1
        docker rm ${cid_value} > /dev/null 2>&1
    fi
    return 0
}

function main() {
    prepare_workspace
    start_docker
    stop_docker
}

main
