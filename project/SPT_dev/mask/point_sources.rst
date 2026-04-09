**************************
Finding sources in SPT maps
**************************

We use ``dory`` from tenki (https://github.com/amaurea/tenki/tree/master) to find sources in SPT maps. 
We first need to project SPT masks and maps to CAR so dory can read these. 
``project_maps_SPT_patch.py`` defines a CAR template around SPT survey and projects 90GHz full maps, ivar (normalized ivar * 1e-2 to mimic a 5uK.arcmin depth, but we should look for raw ivar maps) and masks.
Go in ``python/mask/project_maps_SPT_patch.py`` and modify ``save_dir`` if you want to save maps somewhere in particular, then run:

.. code:: shell
    salloc --nodes 1 --qos interactive --time 00:20:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python mask/project_maps_SPT_patch.py

You can now get dory by cloning ``tenki``.
Before running ``dory``, I had some problems with floating point precision (?) in ``tenki/point_sources/dory.py``, so go to line 119 and change ``> 0`` to ``> 1e-10``.
Otherwise, masks will look very weird and the code will not find any sources.
We can now run ``dory`` (change your path to dory if needed), you can remove "debug" in --output if you trust what I did (I wouldn't):
.. code:: shell
    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=64 srun -n 4 -c 64 --cpu-bind=cores python tenki/point_sources/dory.py find \
        maps/spt/full_095ghz_CAR.fits \
        maps/spt/ivar_095ghz_CAR.fits \
        maps/catalogs \
        --freq 90 \
        --beam /global/cfs/cdirs/sobs/users/tlouis/spt_data/ancillary_products/generally_applicable/beam_c26_cmb_bl_095ghz.txt \
        -R "tile:3000" \
        --pad 60 \
        -m maps/full/pixel_mask_binary_borders_only_CAR_REV.fits \
        --nsigma 3 \
        --output "full,reg,debug" \
        --verbose

You now have a catalog of sources at ``maps/catalogs/cat.{txt,fits}``.
A first try to make masks with it is in ``make_source_mask.py``:
.. code:: shell
    salloc --nodes 1 --qos interactive --time 00:05:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python mask/make_source_mask.py
