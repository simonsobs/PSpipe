Installing PSpipe at Nersc
----------------------------
If you want to use the healpix version of PSpipe you will have to install cfitsio and healpix
Download the latest version of this library

here is the way I have done it:

FC=ifort ./configure --prefix=/global/homes/t/tlouis/local
make
make install

For healpix 
./configure
"enter name of your F90 compiler ()":  ifort 
Then all the default options until :
"enter location of cfitsio library (/usr/local/lib)":
where I used /global/homes/t/tlouis/local/lib

make

Then I added: 
[ -r /global/homes/t/tlouis/.healpix/3_40_Linux/config ] && . /global/homes/t/tlouis/.healpix/3_40_Linux/config
in my .profile.ext
