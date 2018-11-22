SO power spetrum pipeline
----------------------------
A framework for creating the power spectrum pipeline for the Simons Observatory. Documentation is still under development, but details about how to contribute can be found  in [CONTRIBUTING.md](CONTRIBUTING.md).

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

## The SO map class

Ideally we should be able to handle CAR and HEALPIX map.
In order to do so we have written a so map class that can be used to:
read/write/plot/SHT/upgrade/downgrade Healpix and CAR map.
Simple example for how to use this class can be found in test/

```python
from pspy import so_map
m=so_map.read_map('map.fits')
m.info()
m.plot()
```
