Using ``docker``
~~~~~~~~~~~~~~~~

Given the number of requirements, you can use a ``docker`` image already made with the needed
libraries and everything compiled. You should first install `docker
<https://docs.docker.com/install/>`_ for your operating system.


Bash installation
~~~~~~~~~~~~~~~~~

We have written a simple bash script to install the ``PSpipe`` docker and to clone the main
``PSpipe`` libraries.  Just copy the script in a directory where you want to work with pspipe and
run

.. code:: shell

   $ ./run_docker.sh



This will open a new ``bash`` terminal with a full installation of ``PSpipe``, ``pixell``,
``NaMaster``, ``pspy``... For instance, you can start the ``ipython`` interpreter and run the following
``import`` command

.. code:: shell

   $ ipython
   Python 3.6.9 (default, Nov  7 2019, 10:44:02)
   Type 'copyright', 'credits' or 'license' for more information
   IPython 7.11.1 -- An enhanced Interactive Python. Type '?' for help.

   In [1]: import pixell, pymaster, pspy

You can run the python scripts from the tutorials directory of ``PSpipe``.

When you are done with the image, just type ``exit`` and you will go back to your local machine prompt.

Running ``jupyter`` notebook from docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to start a ``jupyter`` server from the ``PSpipe`` image and run it into your web
browser.  Inside the image terminal, you have to start the ``jupyter`` server by typing

.. code:: shell

   $ jupyter notebook --ip 0.0.0.0

Finally open the ``http`` link (something like ``http://127.0.0.1:8888/?token...``) within your web
browser and you should be able to run one of the ``python`` notebook.

Sharing data between the ``docker`` container and the host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything perfomed within the ``/home/pspipe/workspace`` directory will be reflected into the
``/where/to/work_with_pspipe`` on your host machine. You can then share configuration files, source
codes, data files... between the running ``docker`` container and your local machine. Nothing will
be lost after you exit from the ``docker`` container.

Cryptic ``Killed`` message
~~~~~~~~~~~~~~~~~~~~~~~~~~

Docker for Mac limits the resource available to 2Gb of RAM by default, This might cause the code to
crash unexpectedly with a cryptic ``Killed`` message. It can easily be modified, click on the docker
logo (top right of your screen), go in Preferences/Resources and increase the RAM allocated to
Docker.

Youtube video
~~~~~~~~~~~~~

You are not ready for it:  `youtube <https://www.youtube.com/watch?v=LtIuM3pxkng>`_
