Installation
============

Stable release
--------------

To install piel, run this command in your terminal:

.. code:: console

   $ pip install piel

This is the preferred method to install piel, as it will always install
the most recent stable release.

If you don't have `pip <https://pip.pypa.io>`__ installed, this `Python
installation
guide <http://docs.python-guide.org/en/latest/starting/installation/>`__
can guide you through the process.

From sources
------------

The sources for piel can be downloaded from the `Github
repo <https://github.com/daquintero/piel>`__.

You can either clone the public repository:

.. code:: console

   $ git clone git://github.com/daquintero/piel

Or download the
`tarball <https://github.com/daquintero/piel/tarball/master>`__:

.. code:: console

   $ curl -OJL https://github.com/daquintero/piel/tarball/master

Once you have a copy of the source, you can install it with:

.. code:: console

   $ pip install -e .

Developer’s Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

You might also need to run the following commands to run the examples,
documentation, and full environment:

::

   mamba install pandoc
   mamba install jupyterlab jupytext
   pip install -r requirements_dev.txt

Usage
-----

To use piel in a project you can then do:

.. code:: python

   import piel

Installation Environment Verification
-------------------------------------

We have verified the ``piel``, on the latest Ubuntu LTS. You can then run the above
commands and the dependencies should be resolved. In the future, we will
provide a Docker environment. Note that because ``piel`` is a
microservice and the flow depends on multiple packages, the first import
statement during indexing might take a bit of time.

On the first import, the package will create a folder in your home directory called
``.piel``. This folder is used to manage installation requirements and guarantee
reproducible behaviours of the project interactions with the filesystem with the necessary tools.
