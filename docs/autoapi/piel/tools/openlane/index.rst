:py:mod:`piel.tools.openlane`
=============================

.. py:module:: piel.tools.openlane


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   parse/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   defaults/index.rst
   migrate/index.rst
   utils/index.rst
   v1/index.rst
   v2/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.openlane.get_design_from_openlane_migration
   piel.tools.openlane.extract_datetime_from_path
   piel.tools.openlane.find_all_design_runs
   piel.tools.openlane.find_latest_design_run
   piel.tools.openlane.get_gds_path_from_design_run
   piel.tools.openlane.get_design_run_version
   piel.tools.openlane.sort_design_runs
   piel.tools.openlane.check_config_json_exists_openlane_v1
   piel.tools.openlane.check_design_exists_openlane_v1
   piel.tools.openlane.configure_and_run_design_openlane_v1
   piel.tools.openlane.configure_parametric_designs_openlane_v1
   piel.tools.openlane.configure_flow_script_openlane_v1
   piel.tools.openlane.create_parametric_designs_openlane_v1
   piel.tools.openlane.get_design_directory_from_root_openlane_v1
   piel.tools.openlane.get_latest_version_root_openlane_v1
   piel.tools.openlane.read_configuration_openlane_v1
   piel.tools.openlane.write_configuration_openlane_v1
   piel.tools.openlane.filter_timing_sta_files
   piel.tools.openlane.filter_power_sta_files
   piel.tools.openlane.get_all_timing_sta_files
   piel.tools.openlane.get_all_power_sta_files
   piel.tools.openlane.calculate_max_frame_amount
   piel.tools.openlane.calculate_propagation_delay_from_file
   piel.tools.openlane.calculate_propagation_delay_from_timing_data
   piel.tools.openlane.configure_timing_data_rows
   piel.tools.openlane.configure_frame_id
   piel.tools.openlane.filter_timing_data_by_net_name_and_type
   piel.tools.openlane.get_frame_meta_data
   piel.tools.openlane.get_frame_lines_data
   piel.tools.openlane.get_frame_timing_data
   piel.tools.openlane.get_all_timing_data_from_file
   piel.tools.openlane.read_sta_rpt_fwf_file
   piel.tools.openlane.contains_in_lines
   piel.tools.openlane.create_file_lines_dataframe
   piel.tools.openlane.get_file_line_by_keyword
   piel.tools.openlane.read_file_lines
   piel.tools.openlane.get_all_designs_metrics_openlane_v2
   piel.tools.openlane.read_metrics_openlane_v2
   piel.tools.openlane.run_openlane_flow



.. py:function:: get_design_from_openlane_migration(v1: bool = True, design_name_v1: str | None = None, design_directory: str | pathlib.Path | None = None, root_directory_v1: str | pathlib.Path | None = None) -> (str, pathlib.Path)

   This function provides the integration mechanism for easily migrating the interconnection with other toolsets from an OpenLane v1 design to an OpenLane v2 design.

   This function checks if the inputs are to be treated as v1 inputs. If so, and a `design_name` is provided then it will set the `design_directory` to the corresponding `design_name` directory in the corresponding `root_directory_v1 / designs`. If no `root_directory` is provided then it returns `$OPENLANE_ROOT/"<latest>"/. If a `design_directory` is provided then this will always take precedence even with a `v1` flag.

   :param v1: If True, it will migrate from v1 to v2.
   :type v1: bool
   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH. Optional path for v2-based designs.
   :type design_directory: str
   :param root_directory_v1: Root directory of OpenLane v1. If set to None it will return `$OPENLANE_ROOT/"<latest>"`
   :type root_directory_v1: str

   :returns: None


.. py:function:: extract_datetime_from_path(run_path: pathlib.Path) -> str

   Extracts the datetime from a given `run_path` and returns it as a string.


.. py:function:: find_all_design_runs(design_directory: piel.types.PathTypes, run_name: str | None = None) -> list[pathlib.Path]

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory. This function sorts the runs according to the default notations between both `openlane` and `openlane2` run formats.

   If a `run_name` is specified, then the function will return the exact run if it exists. Otherwise, it will return the latest run

   :param design_directory: The path to the design directory
   :type design_directory: PathTypes
   :param run_name: The name of the run to return. Defaults to None.
   :type run_name: str, optional
   :param version: The version of OpenLane to use. Defaults to None.
   :type version: Literal["v1", "v2"], optional

   :raises ValueError: If the run_name is specified but not found in the design_directory

   :returns: A list of pathlib.Path objects corresponding to the runs
   :rtype: list[pathlib.Path]


.. py:function:: find_latest_design_run(design_directory: piel.types.PathTypes, run_name: str | None = None, version: Literal[v1, v2] | None = None) -> (pathlib.Path, str)

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory. This function sorts the runs according to the default notations between both `openlane` and `openlane2` run formats.

   If a `run_name` is specified, then the function will return the exact run if it exists. Otherwise, it will return the latest run.

   :param design_directory: The path to the design directory
   :type design_directory: PathTypes
   :param run_name: The name of the run to return. Defaults to None.
   :type run_name: str, optional
   :param version: The version of the run to return. Defaults to None.
   :type version: Literal["v1", "v2"], optional

   :raises ValueError: If the run_name is specified but not found in the design_directory

   :returns: A tuple of the latest run path and the version
   :rtype: (pathlib.Path, str)


.. py:function:: get_gds_path_from_design_run(design_directory: piel.types.PathTypes, run_directory: piel.types.PathTypes | None = None) -> pathlib.Path

   Returns the path to the final GDS generated by OpenLane.

   :param design_directory: The path to the design directory
   :type design_directory: PathTypes
   :param run_directory: The path to the run directory. Defaults to None. Otherwise gets the latest run.
   :type run_directory: PathTypes, optional

   :returns: The path to the final GDS
   :rtype: pathlib.Path


.. py:function:: get_design_run_version(run_directory: piel.types.PathTypes) -> Literal[v1, v2]

   Returns the version of the design run.


.. py:function:: sort_design_runs(path_list: list[pathlib.Path]) -> dict[str, list[pathlib.Path]]

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory. This function sorts the runs according to the default notations between both `openlane` and `openlane2` run formats.

   :param path_list: A list of pathlib.Path objects corresponding to the runs
   :type path_list: list[pathlib.Path]

   :returns: A dictionary of sorted runs
   :rtype: dict[str, list[pathlib.Path]]


.. py:function:: check_config_json_exists_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> bool

   Checks if a design has a `config.json` file.

   :param design_name: Name of the design.
   :type design_name: str

   :returns: True if `config.json` exists.
   :rtype: config_json_exists(bool)


.. py:function:: check_design_exists_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> bool

   Checks if a design exists in the OpenLane v1 design folder.

   Lists all designs inside the Openlane V1 design root.

   :param design_name: Name of the design.
   :type design_name: str

   :returns: True if design exists.
   :rtype: design_exists(bool)


.. py:function:: configure_and_run_design_openlane_v1(design_name: str, configuration: dict | None = None, root_directory: str | pathlib.Path | None = None) -> None

   Configures and runs an OpenLane v1 design.

   This function does the following:
   1. Check that the design_directory provided is under $OPENLANE_ROOT/<latestversion>/designs
   2. Check if `config.json` has already been provided for this design. If a configuration dictionary is inputted into the function parameters, then it overwrites the default `config.json`.
   3. Create a script directory, a script is written and permissions are provided for it to be executable.
   4. Permit and execute the `openlane_flow.sh` script in the `scripts` directory.

   :param design_name: Name of the design.
   :type design_name: str
   :param configuration: Configuration dictionary.
   :type configuration: dict | None
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: None


.. py:function:: configure_parametric_designs_openlane_v1(design_name: str, parameter_sweep_dictionary: dict, add_id: bool = True) -> list

   For a given `source_design_directory`, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

   :param add_id: Add an ID to the design name. Defaults to True.
   :type add_id: bool
   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param source_design_directory: Source design directory.
   :type source_design_directory: str | pathlib.Path

   :returns: List of configurations to sweep.
   :rtype: configuration_sweep(list)


.. py:function:: configure_flow_script_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> None

   Configures the OpenLane v1 flow script after checking that the design directory exists.

   :param design_directory: Design directory. Defaults to latest OpenLane root.
   :type design_directory: str | pathlib.Path | None

   :returns: None


.. py:function:: create_parametric_designs_openlane_v1(design_name: str, parameter_sweep_dictionary: dict, target_directory: str | pathlib.Path | None = None) -> None

   Takes a OpenLane v1 source directory and creates a parametric combination of these designs.

   :param design_name: Name of the design.
   :type design_name: str
   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param target_directory: Optional target directory.
   :type target_directory: str | pathlib.Path | None

   :returns: None


.. py:function:: get_design_directory_from_root_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> pathlib.Path

   Gets the design directory from the root directory.

   :param design_name: Name of the design.
   :type design_name: str
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: Design directory.
   :rtype: design_directory(pathlib.Path)


.. py:function:: get_latest_version_root_openlane_v1() -> pathlib.Path

   Gets the latest version root of OpenLane v1.


.. py:function:: read_configuration_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> dict

   Reads a `config.json` from a design directory.

   :param design_name: Design name.
   :type design_name: str
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: Configuration dictionary.
   :rtype: configuration(dict)


.. py:function:: write_configuration_openlane_v1(configuration: dict, design_directory: piel.types.PathTypes) -> None

   Writes a `config.json` onto a `design_directory`

   :param configuration: OpenLane configuration dictionary.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None


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


.. py:function:: calculate_max_frame_amount(file_lines_data: pandas.DataFrame)

   Calculate the maximum frame amount based on the frame IDs in the DataFrame

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame

   :returns: Maximum number of frames in the file
   :rtype: maximum_frame_amount (int)


.. py:function:: calculate_propagation_delay_from_file(file_path: str | pathlib.Path)

   Calculate the propagation delay for each frame in the file

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

   :returns: Dictionary containing the propagation delay
   :rtype: propagation_delay (dict)


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


.. py:function:: get_frame_lines_data(file_path: str | pathlib.Path)

   Calculate the timing data for each frame in the file

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

   :returns: DataFrame containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


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


.. py:function:: get_all_timing_data_from_file(file_path: str | pathlib.Path)

   Calculate the timing data for each frame in the file

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

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


.. py:function:: contains_in_lines(file_lines_data: pandas.DataFrame, keyword: str)

   Check if the keyword is contained in the file lines

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame
   :param keyword: Keyword to search for
   :type keyword: str

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: create_file_lines_dataframe(file_lines_raw)

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


.. py:function:: read_file_lines(file_path: str | pathlib.Path)

   Extract lines from the file

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

   :returns: list containing the file lines
   :rtype: file_lines_raw (list)


.. py:function:: get_all_designs_metrics_openlane_v2(output_directory: piel.types.PathTypes, target_prefix: str)

   Returns a dictionary of all the metrics for all the designs in the output directory.

   Usage:

       ```python
       from piel.tools.openlane import get_all_designs_metrics_v2

       metrics = get_all_designs_metrics_v2(
           output_directory="output",
           target_prefix="design",
       )
       ```

   :param output_directory: The path to the output directory.
   :type output_directory: PathTypes
   :param target_prefix: The prefix of the designs to get the metrics for.
   :type target_prefix: str

   :returns: A dictionary of all the metrics for all the designs in the output directory.
   :rtype: dict


.. py:function:: read_metrics_openlane_v2(design_directory: piel.types.PathTypes) -> dict

   Read design metrics from OpenLane v2 run files.

   :param design_directory: Design directory PATH.
   :type design_directory: PathTypes

   :returns: Metrics dictionary.
   :rtype: dict


.. py:function:: run_openlane_flow(configuration: dict | None = None, design_directory: piel.types.PathTypes = '.', parallel_asynchronous_run: bool = False, only_generate_flow_setup: bool = False)

   Runs the OpenLane v2 flow.

   :param configuration: OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: PathTypes
   :param parallel_asynchronous_run: Run the flow in parallel.
   :type parallel_asynchronous_run: bool
   :param only_generate_flow_setup: Only generate the flow setup.
   :type only_generate_flow_setup: bool

   Returns:



