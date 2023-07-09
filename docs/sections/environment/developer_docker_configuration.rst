For Developers
--------------

Note that in the docker container, it is just an environment, it is not
an independent file system. It appears like the only folder that can be
edited from ``iic-osic-tools`` is the ``designs`` folder, maybe I will
see how to disable this in another set of instructions. This means that,
say, you want to install ``piel`` in the default docker environment you
might have to run:

.. code:: shell

   cd $HOME/eda/designs # The default $DESIGN environment
   git clone https://github.com/daquintero/piel.git
