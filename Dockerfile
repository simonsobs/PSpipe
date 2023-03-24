FROM ubuntu:22.04
MAINTAINER Xavier Garrido <xavier.garrido@ijclab.in2p3.fr>

RUN apt-get update && apt-get install -y \
        automake                         \
        build-essential                  \
        emacs                            \
        gfortran                         \
        git                              \
        libcfitsio-dev                   \
        libfftw3-dev                     \
        libgsl-dev                       \
        libchealpix-dev                  \
        libopenmpi-dev                   \
        python3                          \
        python3-pip                      \
        vim                              \
        wget

RUN ln -sfn /usr/bin/python3 /usr/bin/python

RUN useradd -m -U pspipe
USER pspipe
ENV USER pspipe
ENV PATH "/home/pspipe/.local/bin:${PATH}"
WORKDIR /home/pspipe

RUN python3 -m pip install --user --upgrade pip wheel
RUN python3 -m pip install --user --upgrade numpy cython ipython jupyter
RUN python3 -m pip install --user --upgrade astropy astropy-helpers healpy mpi4py numba toml
RUN python3 -m pip install --user --upgrade pysm3
RUN python3 -m pip install --user git+https://github.com/simonsobs/PSpipe.git
