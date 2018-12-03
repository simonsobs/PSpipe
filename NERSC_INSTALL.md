Installing PSpipe at Nersc
----------------------------
If you want to use the healpix version of PSpipe you will have to install cfitsio and healpix. Here is the way I have done it:

First I have downloaded the latest version of these libraries (Healpix_3.40_2018Jun22,cfitsio3450). Then I compiled  cfitsio

```bash
FC=ifort ./configure --prefix=/global/homes/t/tlouis/local
make
make install
```

It was succesful so I moved to Healpix 

```bash
./configure
```

"enter name of your F90 compiler ()":  ifort 

Then I used all the default options until :

"enter location of cfitsio library (/usr/local/lib)":

where I used /global/homes/t/tlouis/local/lib

make

Finally I added: 
[ -r /global/homes/t/tlouis/.healpix/3_40_Linux/config ] && . /global/homes/t/tlouis/.healpix/3_40_Linux/config
in my .profile.ext
