:py:mod:`piel.openlane.parse`
=============================

.. py:module:: piel.openlane.parse

.. autoapi-nested-parse::

   These functions aim to provide functionality to parse data from any OpenLanes v1 design into Python.

   They are ported from the old: github.com/daquintero/porf



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   run_output/index.rst
   sta_rpt/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.openlane.parse.return_path
   piel.openlane.parse.get_files_recursively_in_directory
   piel.openlane.parse.filter_timing_sta_files
   piel.openlane.parse.filter_power_sta_files
   piel.openlane.parse.get_all_timing_sta_files
   piel.openlane.parse.get_all_power_sta_files
   piel.openlane.parse.return_path
   piel.openlane.parse.contains_in_lines
   piel.openlane.parse.read_file_lines
   piel.openlane.parse.get_file_line_by_keyword
   piel.openlane.parse.calculate_max_frame_amount
   piel.openlane.parse.calculate_propagation_delay_from_timing_data
   piel.openlane.parse.calculate_propagation_delay_from_file
   piel.openlane.parse.configure_timing_data_rows
   piel.openlane.parse.configure_frame_id
   piel.openlane.parse.filter_timing_data_by_net_name_and_type
   piel.openlane.parse.get_frame_meta_data
   piel.openlane.parse.get_frame_timing_data
   piel.openlane.parse.get_all_timing_data_from_file
   piel.openlane.parse.read_sta_rpt_fwf_file
   piel.openlane.parse.check_path_exists
   piel.openlane.parse.return_path
   piel.openlane.parse.contains_in_lines
   piel.openlane.parse.create_dataframe
   piel.openlane.parse.get_file_line_by_keyword
   piel.openlane.parse.read_file_lines
   piel.openlane.parse.read_file



.. py:function:: return_path(input_path: str | pathlib.Path) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems.

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: get_files_recursively_in_directory(path: str | pathlib.Path, extension: str = '*')

   Returns a list of files in a directory.

   :param path: Input path.
   :type path: str | pathlib.Path
   :param extension: File extension.
   :type extension: str

   :returns: List of files.
   :rtype: file_list(list)


.. py:function:: filter_timing_sta_files(file_list)

   Filter the timing sta files from the list of files

   :param file_list: List containing the file paths
   :type file_list: list

   :returns: List containing the timing sta files
   :rtype: timing_sta_files (list)


.. py:function:: filter_power_sta_files(file_list)

   Filter the power sta files from the list of files

   :param file_list: List containing the file paths
   :type file_list: list

   :returns: List containing the power sta files
   :rtype: power_sta_files (list)


.. py:function:: get_all_timing_sta_files(run_directory)

   This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

   :param run_directory: The run directory to perform the analysis on. Defaults to None.
   :type run_directory: str, optional

   :returns: List of all the .rpt files in the run directory.
   :rtype: timing_sta_files_list (list)


.. py:function:: get_all_power_sta_files(run_directory)

   This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

   :param run_directory: The run directory to perform the analysis on. Defaults to None.
   :type run_directory: str, optional

   :returns: List of all the .rpt files in the run directory.
   :rtype: power_sta_files_list (list)


.. py:function:: return_path(input_path: str | pathlib.Path) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems.

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: contains_in_lines(file_lines_data: pandas.DataFrame, keyword: str)

   Check if the keyword is contained in the file lines

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame
   :param keyword: Keyword to search for
   :type keyword: str

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: read_file_lines(file)

   Extract lines from the file

   :param file: the opened file

   :returns: list containing the file lines
   :rtype: file_lines_raw (list)


.. py:function:: get_file_line_by_keyword(file_lines_data: pandas.DataFrame, keyword: str, regex: str)

   Extract the data from the file lines using the given keyword and regex

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame
   :param keyword: Keyword to search for
   :type keyword: str
   :param regex: Regex to extract the data
   :type regex: str

   :returns: Dataframe containing the extracted values
   :rtype: extracted_values (pd.DataFrame)


.. py:function:: calculate_max_frame_amount(file_lines_data: pandas.DataFrame)

   Calculate the maximum frame amount based on the frame IDs in the DataFrame

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame

   :returns: Maximum number of frames in the file
   :rtype: maximum_frame_amount (int)


.. py:function:: calculate_propagation_delay_from_timing_data(net_name_in: str, net_name_out: str, timing_data: pandas.DataFrame)

   Calculate the propagation delay between two nets

   :param net_name_in: Name of the input net
   :type net_name_in: str
   :param net_name_out: Name of the output net
   :type net_name_out: str
   :param timing_data: Dataframe containing the timing data
   :type timing_data: pd.DataFrame

   :returns: Dataframe containing the propagation delay
   :rtype: propagation_delay_dataframe (pd.DataFrame)


.. py:function:: calculate_propagation_delay_from_file(file: str | pathlib.Path)

   Calculate the propagation delay for each frame in the file

   :param file: Path to the file
   :type file: str | pathlib.Path

   :returns: Dictionary containing the propagation delay
   :rtype: propagation_delay (dict)


.. py:function:: configure_timing_data_rows(file_lines_data: pandas.DataFrame)

   Identify the timing data lines for each frame and creates a metadata dictionary for frames.

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame

   :returns: Dictionary containing the frame metadata
   :rtype: frame_meta_data (dict)


.. py:function:: configure_frame_id(file_lines_data: pandas.DataFrame)

   Identify the frame delimiters and assign frame ID to each line in the file

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: filter_timing_data_by_net_name_and_type(timing_data: pandas.DataFrame, net_name: str, net_type: str)

   Filter the timing data by net name and type

   :param timing_data: DataFrame containing the timing data
   :type timing_data: pd.DataFrame
   :param net_name: Net name to be filtered
   :type net_name: str
   :param net_type: Net type to be filtered
   :type net_type: str

   :returns: DataFrame containing the timing data
   :rtype: timing_data (pd.DataFrame)


.. py:function:: get_frame_meta_data(file_lines_data)

   Get the frame metadata

   :param file_lines_data: DataFrame containing the file lines
   :type file_lines_data: pd.DataFrame

   :returns: DataFrame containing the start point name
             end_point_name (pd.DataFrame): DataFrame containing the end point name
             path_group_name (pd.DataFrame): DataFrame containing the path group name
             path_type_name (pd.DataFrame): DataFrame containing the path type name
   :rtype: start_point_name (pd.DataFrame)


.. py:function:: get_frame_timing_data(file: str | pathlib.Path, frame_meta_data: dict, frame_id: int = 0)

   Extract the timing data from the file

   :param file: Address of the file
   :type file: str | pathlib.Path
   :param frame_meta_data: Dictionary containing the frame metadata
   :type frame_meta_data: dict
   :param frame_id: Frame ID to be read
   :type frame_id: int

   :returns: DataFrame containing the timing data
   :rtype: timing_data (pd.DataFrame)


.. py:function:: get_all_timing_data_from_file(file: str | pathlib.Path)

   Calculate the timing data for each frame in the file

   :param file: Path to the file
   :type file: str | pathlib.Path

   :returns: Dictionary containing the timing data for each frame
   :rtype: frame_timing_data (dict)


.. py:function:: read_sta_rpt_fwf_file(file: str | pathlib.Path, frame_meta_data: dict, frame_id: int = 0)

   Read the fixed width file and return a DataFrame

   :param file: Address of the file
   :type file: str | pathlib.Path
   :param frame_meta_data: Dictionary containing the frame metadata
   :type frame_meta_data: dict
   :param frame_id: Frame ID to be read
   :type frame_id: int

   :returns: DataFrame containing the file data
   :rtype: file_data (pd.DataFrame)


.. py:function:: check_path_exists(path: str | pathlib.Path, raise_errors: bool = False) -> bool

   Checks if a directory exists.

   :param path: Input path.
   :type path: str | pathlib.Path

   :returns: True if directory exists.
   :rtype: directory_exists(bool)


.. py:function:: return_path(input_path: str | pathlib.Path) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems.

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: contains_in_lines(file_lines_data: pandas.DataFrame, keyword: str)

   Check if the keyword is contained in the file lines

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame
   :param keyword: Keyword to search for
   :type keyword: str

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: create_dataframe(file_lines_raw)

   Create a DataFrame from the raw lines of a file

   :param file_lines_raw: list containing the file lines
   :type file_lines_raw: list

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: get_file_line_by_keyword(file_lines_data: pandas.DataFrame, keyword: str, regex: str)

   Extract the data from the file lines using the given keyword and regex

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame
   :param keyword: Keyword to search for
   :type keyword: str
   :param regex: Regex to extract the data
   :type regex: str

   :returns: Dataframe containing the extracted values
   :rtype: extracted_values (pd.DataFrame)


.. py:function:: read_file_lines(file)

   Extract lines from the file

   :param file: the opened file

   :returns: list containing the file lines
   :rtype: file_lines_raw (list)


.. py:function:: read_file(file_path: str | pathlib.Path)

   Read the file from the given path

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

   :returns: the opened file
   :rtype: file
