SO power spectrum pipeline
----------------------------
A framework for creating the power spectrum pipeline for the Simons Observatory. Documentation is still under development, but details about how to contribute can be found  in [CONTRIBUTING.md](CONTRIBUTING.md).

* Documentation: https://pspipe.readthedocs.io

## Installation

PSpipe will have 3 mains module, pspy, psc, and pslike.
To install `pspy`, just clone this repository and run
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
