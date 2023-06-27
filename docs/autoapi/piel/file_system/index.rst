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
   piel.file_system.create_new_directory
   piel.file_system.delete_path
   piel.file_system.delete_path_list_in_directory
   piel.file_system.get_files_recursively_in_directory
   piel.file_system.permit_script_execution
   piel.file_system.permit_directory_all
   piel.file_system.read_json
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


.. py:function:: create_new_directory(directory_path: str | pathlib.Path) -> None

   Creates a new directory.

   If the parents of the target_directory do not exist, they will be created too.

   :param directory_path: Input path.
   :type directory_path: str | pathlib.Path

   :returns: None


.. py:function:: delete_path(path: str | pathlib.Path) -> None

   Deletes a path.

   :param path: Input path.
   :type path: str | pathlib.Path

   :returns: None


.. py:function:: delete_path_list_in_directory(directory_path: str | pathlib.Path, path_list: list, ignore_confirmation: bool = False, validate_individual: bool = False) -> None

   Deletes a list of files in a directory.

   :param directory_path: Input path.
   :type directory_path: str | pathlib.Path
   :param path_list: List of files.
   :type path_list: list
   :param ignore_confirmation: Ignore confirmation. Default: False.
   :type ignore_confirmation: bool
   :param validate_individual: Validate individual files. Default: False.
   :type validate_individual: bool

   :returns: None


.. py:function:: get_files_recursively_in_directory(path: str | pathlib.Path, extension: str = '*')

   Returns a list of files in a directory.

   :param path: Input path.
   :type path: str | pathlib.Path
   :param extension: File extension.
   :type extension: str

   :returns: List of files.
   :rtype: file_list(list)


.. py:function:: permit_script_execution(script_path: str | pathlib.Path) -> None

   Permits the execution of a script.

   :param script_path: Script path.
   :type script_path: str

   :returns: None


.. py:function:: permit_directory_all(directory_path: str | pathlib.Path) -> None

   Permits a directory to be read, written and executed. Use with care as it can be a source for security issues.

   :param directory_path: Input path.
   :type directory_path: str | pathlib.Path

   :returns: None


.. py:function:: read_json(path: str | pathlib.Path) -> dict

   Reads a JSON file.

   :param path: Input path.
   :type path: str | pathlib.Path

   :returns: JSON data.
   :rtype: json_data(dict)


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
