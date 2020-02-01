FROM ubuntu:18.04
MAINTAINER Xavier Garrido <xavier.garrido@lal.in2p3.fr>

RUN apt-get update && apt-get install -y \
        automake                         \
        build-essential                  \
        gfortran                         \
        git                              \
        libcfitsio-dev                   \
        libfftw3-dev                     \
        libgsl-dev                       \
        libchealpix-dev                  \
        python3                          \
        python3-pip                      \
        wget

RUN ln -sfn /usr/bin/python3 /usr/bin/python

RUN useradd -m -U pspipe
USER pspipe
ENV USER pspipe
ENV PATH "/home/pspipe/.local/bin:${PATH}"
WORKDIR /home/pspipe

RUN python3 -m pip install --user --upgrade pip numpy cython ipython jupyter
RUN git clone https://github.com/simonsobs/PSpipe.git && cd PSpipe && python3 -m pip install --user .
