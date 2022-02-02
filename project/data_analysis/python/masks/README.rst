*******************
Generation of masks
*******************

This directory holds two scripts:

- the ``generate_point_sources_mask.py`` script generates a point sources mask file given a
  configuration file

- the ``generate_patches_mask.py`` script generates a set of patch masks *i.e.* region of the sky
  mask

- the ``generate_galactic_mask.py`` script generates galactic mask in equatorial coordinates and
  CAR pixelisation given an HEALPIX mask. For instance, it will project Planck galactic masks onto
  ACT survey

For instance, you can run both scripts with the ``masks.dict`` dictionnary file as follow

.. code:: shell

    python generate_point_sources_mask.py masks.dict
