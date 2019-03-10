include ../compile_opts/$(PSPY_COMP).mk

all: cov_fortran.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c ../wigner3j/wigner3j_sub.f -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK) skip: map_border calc_weights

clean:
	rm -rf *.pyc *.so *.mod *.pyf
