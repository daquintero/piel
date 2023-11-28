:py:mod:`piel.cli.develop`
==========================

.. py:module:: piel.cli.develop


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.cli.develop.develop
   piel.cli.develop.build_documentation
   piel.cli.develop.generate_poetry2nix_flake
   piel.cli.develop.build_piel_cachix_command



.. py:function:: develop()

   Development related commands.


.. py:function:: build_documentation(args=None)

   Verifies and builds the documentation.


.. py:function:: generate_poetry2nix_flake(args=None)

   Generates the poetry2nix flakes file. Requires nix to be installed.

   Returns:



.. py:function:: build_piel_cachix_command(args=None)

   Enters the custom piel nix environment with all the supported tools installed and configured packages.
   Runs the nix-shell command on the piel/environment/nix/ directory.


