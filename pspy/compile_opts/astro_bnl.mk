# export LAPACK_LINK = -L$(MKLPATH) -lmkl_rt -lpthread -lm
# export OMP_LINK    = -liomp5
# export FFLAGS      = -openmp -Ofast -fPIC -xhost -DUSE_MPI # -vec-report -opt-report
# export FSAFE       = -openmp -O3 -fPIC -xhost -DUSE_MPI
# export FC          = mpif90
# export F2PY        = f2py
# export F2PYCOMP    = intelem
# export PYTHON      = python
# export SED         = sed
# #export CC          = mpiCC
# export CC          = mpicc


export LAPACK_LINK = -llapack
export OMP_LINK    = -lgomp
export FFLAGS      = -fopenmp -Ofast -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs
#export FFLAGS      = -fopenmp -O0 -fbounds-check -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs
export FSAFE       = -fopenmp -O3 -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs
export FC          = mpif90
export F2PY        = f2py
export F2PYCOMP    = gfortran
export PYTHON      = python
export SED         = sed
export CC          = mpicc
