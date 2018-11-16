export LAPACK_LINK = -llapack
export OMP_LINK    = -lgomp
export FFLAGS      = -fopenmp -Ofast -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs
#export FFLAGS      = -fopenmp -O0 -fbounds-check -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs
export FSAFE       = -fopenmp -O3 -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs
export FC          = gfortran
export F2PY        = f2py2
export F2PYCOMP    = gfortran
export PYTHON      = python2
export SED         = sed
export CC          = gcc
