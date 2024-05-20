:py:mod:`piel.cli.environment.nix`
==================================

.. py:module:: piel.cli.environment.nix


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.cli.environment.nix.activate_openlane_nix
   piel.cli.environment.nix.activate_piel_nix
   piel.cli.environment.nix.install_nix
   piel.cli.environment.nix.install_openlane



.. py:function:: activate_openlane_nix(openlane2_directory: pathlib.Path = default_openlane2_directory)

   CLI that installs both the openlane2 python interface and the OpenROAD binaries.


.. py:function:: activate_piel_nix(openlane2_directory: pathlib.Path = default_openlane2_directory)

   Enters the custom piel nix environment with all the supported tools installed and configured packages.
   Runs the nix-shell command on the piel/environment/nix/ directory.


.. py:function:: install_nix()

   Installs the nix package manager.


.. py:function:: install_openlane(openlane2_directory: pathlib.Path = default_openlane2_directory)

   CLI that installs both the openlane2 python interface and the OpenROAD binaries.
