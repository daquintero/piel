:py:mod:`piel.cli`
==================

.. py:module:: piel.cli


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   environment/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   core/index.rst
   develop/index.rst
   main/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.cli.get_python_install_directory
   piel.cli.main
   piel.cli.get_piel_install_directory
   piel.cli.return_path
   piel.cli.echo_and_check_subprocess
   piel.cli.get_python_install_directory
   piel.cli.main
   piel.cli.develop
   piel.cli.build_documentation
   piel.cli.generate_poetry2nix_flake
   piel.cli.build_piel_cachix_command
   piel.cli.main
   piel.cli.append_to_bashrc_if_does_not_exist
   piel.cli.echo_and_run_subprocess
   piel.cli.echo_and_check_subprocess
   piel.cli.get_python_install_directory
   piel.cli.get_piel_home_directory
   piel.cli.install_nix
   piel.cli.install_openlane
   piel.cli.activate_openlane_nix
   piel.cli.activate_piel_nix
   piel.cli.create_and_activate_venv



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.cli.default_openlane2_directory


.. py:function:: get_python_install_directory()

   Gets the piel installation directory.

   :returns: The piel installation directory.
   :rtype: pathlib.Path


.. py:function:: main(args=None)

   CLI Interface for piel There are available many helper commands to help you set up your
   environment and design your projects.


.. py:function:: get_piel_install_directory()

   Gets the piel installation directory.


.. py:function:: return_path(input_path: piel.types.PathTypes, as_piel_module: bool = False) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems. When the `as_piel_module` flag is
   enabled, it will analyse whether the input path can be treated as a piel module, and treat the returned path as a
   module would be treated. This comes useful when analysing data generated in this particular structure accordingly.

   Usage:

       return_path('path/to/file')

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: echo_and_check_subprocess(command: list, **kwargs)

   Runs a subprocess and prints the command. Raises an exception if the subprocess fails.

   :param command:
   :param \*\*kwargs:

   Returns:



.. py:function:: get_python_install_directory()

   Gets the piel installation directory.

   :returns: The piel installation directory.
   :rtype: pathlib.Path


.. py:function:: main(args=None)

   CLI Interface for piel There are available many helper commands to help you set up your
   environment and design your projects.


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


.. py:function:: main(args=None)

   CLI Interface for piel There are available many helper commands to help you set up your
   environment and design your projects.


.. py:function:: append_to_bashrc_if_does_not_exist(line: str)

   Appends a line to .bashrc if it does not exist.

   :param line:

   Returns:



.. py:data:: default_openlane2_directory



.. py:function:: echo_and_run_subprocess(command: list, **kwargs)

   Runs a subprocess and prints the command.

   :param command:
   :param \*\*kwargs:

   Returns:



.. py:function:: echo_and_check_subprocess(command: list, **kwargs)

   Runs a subprocess and prints the command. Raises an exception if the subprocess fails.

   :param command:
   :param \*\*kwargs:

   Returns:



.. py:function:: get_python_install_directory()

   Gets the piel installation directory.

   :returns: The piel installation directory.
   :rtype: pathlib.Path


.. py:function:: get_piel_home_directory()

   Gets the piel home directory.

   :returns: The piel home directory.
   :rtype: pathlib.Path


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
