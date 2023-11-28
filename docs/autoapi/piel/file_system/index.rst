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
   piel.file_system.copy_example_design
   piel.file_system.create_new_directory
   piel.file_system.create_piel_home_directory
   piel.file_system.delete_path
   piel.file_system.delete_path_list_in_directory
   piel.file_system.get_files_recursively_in_directory
   piel.file_system.get_id_map_directory_dictionary
   piel.file_system.get_top_level_script_directory
   piel.file_system.list_prefix_match_directories
   piel.file_system.permit_script_execution
   piel.file_system.permit_directory_all
   piel.file_system.read_json
   piel.file_system.rename_file
   piel.file_system.rename_files_in_directory
   piel.file_system.replace_string_in_file
   piel.file_system.replace_string_in_directory_files
   piel.file_system.return_path
   piel.file_system.run_script
   piel.file_system.write_file



.. py:function:: check_path_exists(path: piel.config.piel_path_types, raise_errors: bool = False) -> bool

   Checks if a directory exists.

   :param path: Input path.
   :type path: piel_path_types

   :returns: True if directory exists.
   :rtype: directory_exists(bool)


.. py:function:: check_example_design(design_name: str = 'simple_design', designs_directory: piel.config.piel_path_types | None = None) -> bool

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

   :param design_name: Name of the design to check.
   :type design_name: str
   :param designs_directory: Directory that contains the DESIGNS environment flag.
   :type designs_directory: piel_path_types
   :param # TODO:

   :returns: None


.. py:function:: copy_source_folder(source_directory: piel.config.piel_path_types, target_directory: piel.config.piel_path_types) -> None

   Copies the files from a source_directory to a target_directory

   :param source_directory: Source directory.
   :type source_directory: piel_path_types
   :param target_directory: Target directory.
   :type target_directory: piel_path_types

   :returns: None


.. py:function:: copy_example_design(project_source: Literal[piel, openlane] = 'piel', example_name: str = 'simple_design', target_directory: piel.config.piel_path_types = None, target_project_name: Optional[str] = None) -> None

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

   :param project_source: Source of the project.
   :type project_source: str
   :param example_name: Name of the example design.
   :type example_name: str
   :param target_directory: Target directory.
   :type target_directory: piel_path_types
   :param target_project_name: Name of the target project.
   :type target_project_name: str

   :returns: None


.. py:function:: create_new_directory(directory_path: str | pathlib.Path, overwrite: bool = False) -> bool

   Creates a new directory.

   If the parents of the target_directory do not exist, they will be created too.

   :param overwrite: Overwrite directory if it already exists.
   :param directory_path: Input path.
   :type directory_path: str | pathlib.Path

   :returns: None


.. py:function:: create_piel_home_directory() -> None

   Creates the piel home directory.

   :returns: None


.. py:function:: delete_path(path: str | pathlib.Path) -> None

   Deletes a path.

   :param path: Input path.
   :type path: str | pathlib.Path

   :returns: None


.. py:function:: delete_path_list_in_directory(directory_path: piel.config.piel_path_types, path_list: list, ignore_confirmation: bool = False, validate_individual: bool = False) -> None

   Deletes a list of files in a directory.

   Usage:

   ```python
   delete_path_list_in_directory(
       directory_path=directory_path, path_list=path_list, ignore_confirmation=True
   )
   ```

   :param directory_path: Input path.
   :type directory_path: piel_path_types
   :param path_list: List of files.
   :type path_list: list
   :param ignore_confirmation: Ignore confirmation. Default: False.
   :type ignore_confirmation: bool
   :param validate_individual: Validate individual files. Default: False.
   :type validate_individual: bool

   :returns: None


.. py:function:: get_files_recursively_in_directory(path: piel.config.piel_path_types, extension: str = '*')

   Returns a list of files in a directory.

   Usage:

       get_files_recursively_in_directory('path/to/directory', 'extension')

   :param path: Input path.
   :type path: piel_path_types
   :param extension: File extension.
   :type extension: str

   :returns: List of files.
   :rtype: file_list(list)


.. py:function:: get_id_map_directory_dictionary(path_list: list[piel.config.piel_path_types], target_prefix: str)

   Returns a dictionary of ids to directories.

   Usage:

       get_id_to_directory_dictionary(path_list, target_prefix)

   :param path_list: List of paths.
   :type path_list: list[piel_path_types]
   :param target_prefix: Target prefix.
   :type target_prefix: str

   :returns: Dictionary of ids to directories.
   :rtype: id_dict(dict)


.. py:function:: get_top_level_script_directory() -> pathlib.Path

   Returns the top level script directory whenever this file is run. This is useful when we want to know the
   location of the script that is being executed at the top level, maybe in order to create relative directories of
   find relevant files.

   :returns: Top level script directory.
   :rtype: top_level_script_directory(pathlib.Path)


.. py:function:: list_prefix_match_directories(output_directory: piel.config.piel_path_types, target_prefix: str)

   Returns a list of directories that match a prefix.

   Usage:

       list_prefix_match_directories('path/to/directory', 'prefix')

   :param output_directory: Output directory.
   :type output_directory: piel_path_types
   :param target_prefix: Target prefix.
   :type target_prefix: str

   :returns: List of directories.
   :rtype: matching_dirs(list)


.. py:function:: permit_script_execution(script_path: piel.config.piel_path_types) -> None

   Permits the execution of a script.

   Usage:

       permit_script_execution('path/to/script')

   :param script_path: Script path.
   :type script_path: piel_path_types

   :returns: None


.. py:function:: permit_directory_all(directory_path: piel.config.piel_path_types) -> None

   Permits a directory to be read, written and executed. Use with care as it can be a source for security issues.

   Usage:

       permit_directory_all('path/to/directory')

   :param directory_path: Input path.
   :type directory_path: piel_path_types

   :returns: None


.. py:function:: read_json(path: piel.config.piel_path_types) -> dict

   Reads a JSON file.

   Usage:

       read_json('path/to/file.json')

   :param path: Input path.
   :type path: piel_path_types

   :returns: JSON data.
   :rtype: json_data(dict)


.. py:function:: rename_file(match_file_path: piel.config.piel_path_types, renamed_file_path: piel.config.piel_path_types) -> None

   Renames a file.

   Usage:

       rename_file('path/to/match_file', 'path/to/renamed_file')

   :param match_file_path: Input path.
   :type match_file_path: piel_path_types
   :param renamed_file_path: Input path.
   :type renamed_file_path: piel_path_types

   :returns: None


.. py:function:: rename_files_in_directory(target_directory: piel.config.piel_path_types, match_string: str, renamed_string: str) -> None

   Renames all files in a directory.

   Usage:

       rename_files_in_directory('path/to/directory', 'match_string', 'renamed_string')

   :param target_directory: Input path.
   :type target_directory: piel_path_types
   :param match_string: String to match.
   :type match_string: str
   :param renamed_string: String to replace.
   :type renamed_string: str

   :returns: None


.. py:function:: replace_string_in_file(file_path: piel.config.piel_path_types, match_string: str, replace_string: str)

   Replaces a string in a file.

   Usage:

       replace_string_in_file('path/to/file', 'match_string', 'replace_string')

   :param file_path: Input path.
   :type file_path: piel_path_types
   :param match_string: String to match.
   :type match_string: str
   :param replace_string: String to replace.
   :type replace_string: str

   :returns: None


.. py:function:: replace_string_in_directory_files(target_directory: piel.config.piel_path_types, match_string: str, replace_string: str)

   Replaces a string in all files in a directory.

   Usage:

       replace_string_in_directory_files('path/to/directory', 'match_string', 'replace_string')

   :param target_directory: Input path.
   :type target_directory: piel_path_types
   :param match_string: String to match.
   :type match_string: str
   :param replace_string: String to replace.
   :type replace_string: str

   :returns: None


.. py:function:: return_path(input_path: piel.config.piel_path_types, as_piel_module: bool = False) -> pathlib.Path

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


.. py:function:: run_script(script_path: piel.config.piel_path_types) -> None

   Runs a script on the filesystem `script_path`.

   :param script_path: Script path.
   :type script_path: piel_path_types

   :returns: None


.. py:function:: write_file(directory_path: piel.config.piel_path_types, file_text: str, file_name: str) -> None

   Records a `script_name` in the `scripts` project directory.

   :param directory_path: Design directory.
   :type directory_path: piel_path_types
   :param file_text: Script to write.
   :type file_text: str
   :param file_name: Name of the script.
   :type file_name: str

   :returns: None


