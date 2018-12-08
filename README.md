SO power spectrum pipeline
----------------------------
A framework for creating the power spectrum pipeline for the Simons Observatory. Documentation is still under development, but details about how to contribute can be found  in [CONTRIBUTING.md](CONTRIBUTING.md).

## Installation

PSpipe will have 3 mains module, pspy, psc, and pslike.

The healpix version of PSpipe has the following dependencies: cfitsio and HEALPix.

A guide to install these libraries at NERSC is provided in [NERSC_INSTALL.md](NERSC_INSTALL.md)

To install `pspy`, just clone this repository and first compile fortran codes

```
  cd <path to pspy>/pspy
  export PSPY_COMP=<Compiler File> ! ex) export PSPY_COMP=nersc_cori
                                    ! Check pspy/compile_opts for more options
  make
```

Once fortran code is compiled. Go to the root of `pspy` and run

```bash
python setup.py install
```

(add `--user` if you don't have permissions, which is probably the case at e.g. NERSC).

Once installed, you can test the installation by running 
```bash
python test/test_projection.py
```

which project a HEALPIX CMB map into a CAR template.

## The SO map class

We should be able to work with both CAR and HEALPIX pixellisation.
We have written a so map class that can be used to: read/write/plot/SHT/upgrade/downgrade Healpix and CAR maps (using healpy and pixell functionalities).
Simple examples for how to use this class can be found in test/

```python
from pspy import so_map
m=so_map.read_map('map.fits') #map.fits can be a HEALPIX or a CAR map
m.info()
m.plot()
```

## Testing the code

Have a look at the test folder and try to run the scripts.

test_io.py : how to read/write/plot so map

test_projection.py: test of projection healpix to CAR

test_spectra_spin0.py: test of generation of spin0  spectra

test_spectra_spin0and2.py: test of generation of spin0 and spin2 spectra

test_pspy_namaster_spin0.py: comparison pspy / namaster for spin 0 fields (require installing namaster)

test_pspy_namaster_spin0and2.py: comparison pspy / namaster for spin 0 and 2 fields (require installing namaster)




