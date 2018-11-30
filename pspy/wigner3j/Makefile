include ../compile_opts/$(PSPY_COMP).mk

all: wigner3j_sub.so

%.so: %.f
	echo $(basename $<)	
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK) skip: map_border calc_weights

clean:
	rm -rf *.pyc *.so *.mod *.pyf
