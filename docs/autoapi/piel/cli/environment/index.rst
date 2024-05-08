:py:mod:`piel.cli.environment`
==============================

.. py:module:: piel.cli.environment


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   environment/index.rst
   nix/index.rst
   venv/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.cli.environment.install_nix
   piel.cli.environment.install_openlane
   piel.cli.environment.activate_openlane_nix
   piel.cli.environment.activate_piel_nix
   piel.cli.environment.create_and_activate_venv



.. py:function:: install_nix()

   Installs the nix package manager.


.. py:function:: install_openlane(openlane2_directory: pathlib.Path = default_openlane2_directory)

   CLI that installs both the openlane2 python interface and the OpenROAD binaries.


.. py:function:: activate_openlane_nix(openlane2_directory: pathlib.Path = default_openlane2_directory)

   CLI that installs both the openlane2 python interface and the OpenROAD binaries.


.. py:function:: activate_piel_nix(openlane2_directory: pathlib.Path = default_openlane2_directory)

   Enters the custom piel nix environment with all the supported tools installed and configured packages.
   Runs the nix-shell command on the piel/environment/nix/ directory.


.. py:function:: create_and_activate_venv() -> None

   Creates and activates the piel virtual environment.

   :returns: None.
   :rtype: None
