:py:mod:`piel.cli.utils`
========================

.. py:module:: piel.cli.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.cli.utils.append_to_bashrc_if_does_not_exist
   piel.cli.utils.echo_and_run_subprocess
   piel.cli.utils.echo_and_check_subprocess
   piel.cli.utils.get_python_install_directory
   piel.cli.utils.get_piel_home_directory



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.cli.utils.default_openlane2_directory


.. py:data:: default_openlane2_directory



.. py:function:: append_to_bashrc_if_does_not_exist(line: str)

   Appends a line to .bashrc if it does not exist.

   :param line:

   Returns:



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
