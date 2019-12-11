SO power spectrum pipeline
----------------------------
A framework for creating the power spectrum pipeline for the Simons Observatory. Details about how to contribute can be found  in [CONTRIBUTING.md](CONTRIBUTING.md).

## Installation

The SO power spectrum pipeline will have 3 mains modules:

-a python power spectrum code: pspy

-a C power spectrum code: namaster

-a likelihood code: pslike


PSpipe requires pixell:  https://github.com/simonsobs/pixell .

We have simplified the installation of `pspy`, just clone this repository, go to the root of `pspy` and run

```bash
python setup.py install
```

(add `--user` if you don't have permissions, which is probably the case at e.g. NERSC).

It's very difficult to make the installation work on any computer (and we are not software engineer) so if you have an issue, please contact louis@lal.in2p3.fr or dongwon.han@stonybrook.edu and we will try to help.

Few possible problems:

Due to the way pixell works, you don't want to use the code on login node, to test things, go for example on interactive node, and export the number of OMP cores: 

salloc -N 1 -q debug -C haswell -t 00:30:00 -L SCRATCH

export OMP_NUM_THREADS=64

before running any script.

We provide an example on NERSC .bashrc.ext in this folder that can be used if you experienced any issue.


to install namaster, please follow the instruction in:  https://github.com/LSSTDESC/NaMaster


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

## How to learn to use the code/ test the code 

Have a look at the test folder and try to run the scripts.

test_io.py : how to read/write/plot so map

test_projection.py: test of projection healpix to CAR

test_spectra_spin0.py: test of generation of spin0  spectra for HEALPIX and CAR pixellisation 

test_spectra_spin0and2.py: test of generation of spin0 and spin2 spectra for HEALPIX and CAR pixellisation

test_pspy_namaster_spin0.py: comparison pspy / namaster for spin 0 fields (require installing namaster)

test_pspy_namaster_spin0and2.py: comparison pspy / namaster for spin 0 and 2 fields (require installing namaster)




