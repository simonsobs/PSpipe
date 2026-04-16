**************************
Finding sources in SPT maps
**************************

We use ``dory`` from tenki (https://github.com/amaurea/tenki/tree/master) to find sources in SPT maps.
We first need to project SPT masks and maps to CAR so dory can read these.
``project_maps_SPT_patch.py`` defines a CAR template around SPT survey and projects the SPT full maps, ivar and masks.

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python project_maps_SPT_patch.py mask.dict

.. code:: shell

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu
    sh spt_find_source.sh

The code is not working very well (it gets lot of duplicate around super bright sources) but that is the best we have

.. code:: shell

    salloc --nodes 1 --qos interactive --time  04:00:00 --constraint cpu
    sh spt_find_source.sh


You can then generate the mask and check if it does what you want

.. code:: shell

    salloc --nodes 1 --qos interactive --time  01:00:00 --constraint cpu
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python generate_source_mask.py mask.dict
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python check_source_mask.py mask.dict
