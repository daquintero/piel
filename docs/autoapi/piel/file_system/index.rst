:py:mod:`piel.file_system`
==========================

.. py:module:: piel.file_system


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.file_system.check_path_exists
   piel.file_system.check_example_design
   piel.file_system.copy_source_folder
   piel.file_system.permit_script_execution
   piel.file_system.return_path
   piel.file_system.run_script
   piel.file_system.setup_example_design
   piel.file_system.write_script



.. py:function:: check_path_exists(path: str | pathlib.Path, raise_errors: bool = False) -> bool

   Checks if a directory exists.

   :param path: Input path.
   :type path: str | pathlib.Path

   :returns: True if directory exists.
   :rtype: directory_exists(bool)


.. py:function:: check_example_design(design_name: str | pathlib.Path = 'simple_design') -> bool

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

   :param design_name: Name of the design to check.
   :type design_name: str

   :returns: None


.. py:function:: copy_source_folder(source_directory: str | pathlib.Path, target_directory: str | pathlib.Path) -> None

   Copies the files from a source_directory to a target_directory

   :param source_directory: Source directory.
   :type source_directory: str
   :param target_directory: Target directory.
   :type target_directory: str

   :returns: None


.. py:function:: permit_script_execution(script_path: str | pathlib.Path) -> None

   Permits the execution of a script.

   :param script_path: Script path.
   :type script_path: str

   :returns: None


.. py:function:: return_path(input_path: str | pathlib.Path) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems.

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: run_script(script_path: str | pathlib.Path) -> None

   Runs a script on the filesystem `script_path`.

   :param script_path: Script path.
   :type script_path: str

   :returns: None


.. py:function:: setup_example_design(project_source: Literal[piel, openlane] = 'piel', example_name: str = 'simple_design') -> None

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

   :param project_source: Source of the project.
   :type project_source: str
   :param example_name: Name of the example design.
   :type example_name: str

   :returns: None


.. py:function:: write_script(directory_path: str | pathlib.Path, script: str, script_name: str) -> None

   Records a `script_name` in the `scripts` project directory.

   :param directory_path: Design directory.
   :type directory_path: str
   :param script: Script to write.
   :type script: str
   :param script_name: Name of the script.
   :type script_name: str

   :returns: None
