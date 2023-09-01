Linux - Docker Setup Instructions
----------------------------------------------------

We aim for ``piel`` to be automatically installed with all dependencies in the docker environment provided by
`IIC-OSIC-TOOLS <https://github.com/iic-jku/iic-osic-tools>`__. This is an `issue in progress <https://github.com/iic-jku/iic-osic-tools/issues/14>`__

Start your docker environment accordingly running, for example:

.. code:: shell

   cd iic-osic-tools
   ./start_vnc.sh

Find out which docker instance is running:

.. code:: shell

   docker ps

You can start a bash terminal in the correct docker environment by
running:

.. code:: shell

   docker exec -it <yourdockercontainername> bash

You can explore the environment a little bit just between running ``ls``
and ``cd`` commands. If it has started from the default installation you
should have started under the ``foss/designs`` folder. It is here that
we will load our initial design for evaluation and interconnectivity.

Now, we can begin and follow the other tutorials.

Then go into your `localhost:8888 <http://localhost:8888>`__ to access
Jupyter Lab directly from your Chrome notebook.
