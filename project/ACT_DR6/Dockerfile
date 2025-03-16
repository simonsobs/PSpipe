FROM ubuntu:22.04

WORKDIR /opt

RUN <<EOF
    apt-get update
    apt-get install --yes \
        build-essential   \
        gfortran          \
        git               \
        libgsl-dev        \
        python3-dev       \
        python3-pip       \
        python-is-python3 \
        python3-venv      \
        wget
    apt-get clean all
EOF

# Install manually and not using apt packages see
# https://docs.nersc.gov/development/containers/shifter/how-to-use/#using-mpi-in-shifter
ARG mpich=4.0.2
ARG mpich_prefix=mpich-$mpich
RUN <<EOF
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz
    tar xvzf $mpich_prefix.tar.gz
    cd $mpich_prefix
    ./configure FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch
    make -j 16 && make install && make clean
    cd ..
    rm -rf $mpich_prefix $mpich_prefix.tar.gz
EOF

RUN /sbin/ldconfig

RUN <<EOF
    wget https://portal.nersc.gov/cfs/sobs/users/xgarrido/setup.sh
    sed -i                                           \
        -e 's#git@github.com:#https://github.com/#g' \
        -e '/module load\|module swap\|MPICC/d'      \
        -e '/ACTCollaboration/d'                     \
        setup.sh
    /bin/bash -c "source setup.sh"
EOF

ENV VIRTUAL_ENV /opt/pyenv/dr6
ENV PATH /opt/pyenv/dr6/bin:$PATH

# Use this specific version, others seem not to work
RUN python -m pip install mpi4py==3.1.3
