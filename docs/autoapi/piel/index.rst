:py:mod:`piel`
==============

.. py:module:: piel

.. autoapi-nested-parse::

   Top-level package for piel.



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   cocotb/index.rst
   components/index.rst
   integration/index.rst
   models/index.rst
   openlane/index.rst
   sax/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   cli/index.rst
   config/index.rst
   defaults/index.rst
   file_system/index.rst
   parametric/index.rst
   project_structure/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.check_cocotb_testbench_exists
   piel.configure_cocotb_simulation
   piel.run_cocotb_simulation
   piel.check_path_exists
   piel.check_example_design
   piel.copy_source_folder
   piel.create_new_directory
   piel.delete_path
   piel.delete_path_list_in_directory
   piel.get_files_recursively_in_directory
   piel.permit_directory_all
   piel.permit_script_execution
   piel.setup_example_design
   piel.read_json
   piel.return_path
   piel.run_script
   piel.write_script
   piel.create_gdsfactory_component_from_openlane
   piel.get_design_directory_from_root_openlane_v1
   piel.return_path
   piel.get_design_from_openlane_migration
   piel.find_design_run
   piel.check_config_json_exists_openlane_v1
   piel.check_design_exists_openlane_v1
   piel.configure_and_run_design_openlane_v1
   piel.configure_parametric_designs_openlane_v1
   piel.configure_flow_script_openlane_v1
   piel.create_parametric_designs_openlane_v1
   piel.get_latest_version_root_openlane_v1
   piel.read_configuration_openlane_v1
   piel.write_configuration_openlane_v1
   piel.get_files_recursively_in_directory
   piel.filter_timing_sta_files
   piel.filter_power_sta_files
   piel.get_all_timing_sta_files
   piel.get_all_power_sta_files
   piel.contains_in_lines
   piel.read_file_lines
   piel.get_file_line_by_keyword
   piel.create_file_lines_dataframe
   piel.calculate_max_frame_amount
   piel.calculate_propagation_delay_from_timing_data
   piel.calculate_propagation_delay_from_file
   piel.configure_timing_data_rows
   piel.configure_frame_id
   piel.filter_timing_data_by_net_name_and_type
   piel.get_frame_meta_data
   piel.get_frame_lines_data
   piel.get_frame_timing_data
   piel.get_all_timing_data_from_file
   piel.read_sta_rpt_fwf_file
   piel.check_path_exists
   piel.read_file
   piel.run_openlane_flow
   piel.single_parameter_sweep
   piel.multi_parameter_sweep



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.delete_simulation_output_files
   piel.numerical_solver
   piel.nso
   piel.test_spm_open_lane_configuration
   piel.example_open_lane_configuration
   piel.__author__
   piel.__email__
   piel.__version__


.. py:function:: check_cocotb_testbench_exists(design_directory: str | pathlib.Path) -> bool

   Checks if a cocotb testbench exists in the design directory.

   :param design_directory: Design directory.
   :type design_directory: str | pathlib.Path

   :returns: True if cocotb testbench exists.
   :rtype: cocotb_testbench_exists(bool)


.. py:function:: configure_cocotb_simulation(design_directory: str | pathlib.Path, simulator: Literal[icarus, verilator], top_level_language: Literal[verilog, vhdl], top_level_verilog_module: str, test_python_module: str, design_sources_list: list | None = None)

   Writes a cocotb makefile.

   If no design_sources_list is provided then it adds all the design sources under the `src` folder.

   In the form
   .. code-block::

       #!/bin/sh
       # Makefile
       # defaults
       SIM ?= icarus
       TOPLEVEL_LANG ?= verilog

       # Note we need to include the test script to the PYTHONPATH
       export PYTHONPATH =

       VERILOG_SOURCES += $(PWD)/my_design.sv
       # use VHDL_SOURCES for VHDL files

       # TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
       TOPLEVEL := my_design

       # MODULE is the basename of the Python test file
       MODULE := test_my_design

       # include cocotb's make rules to take care of the simulator setup
       include $(shell cocotb-config --makefiles)/Makefile.sim


   :param design_directory: The directory where the design is located.
   :type design_directory: str | pathlib.Path
   :param simulator: The simulator to use.
   :type simulator: Literal["icarus", "verilator"]
   :param top_level_language: The top level language.
   :type top_level_language: Literal["verilog", "vhdl"]
   :param top_level_verilog_module: The top level verilog module.
   :type top_level_verilog_module: str
   :param test_python_module: The test python module.
   :type test_python_module: str
   :param design_sources_list: A list of design sources. Defaults to None.
   :type design_sources_list: list | None, optional

   :returns: None


.. py:data:: delete_simulation_output_files



.. py:function:: run_cocotb_simulation(design_directory: str) -> subprocess.CompletedProcess

   Equivalent to running the cocotb makefile
   .. code-block::

       make

   :param design_directory: The directory where the design is located.
   :type design_directory: str

   :returns: The subprocess.CompletedProcess object.
   :rtype: subprocess.CompletedProcess


.. py:data:: numerical_solver



.. py:data:: nso



.. py:data:: test_spm_open_lane_configuration



.. py:data:: example_open_lane_configuration



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


.. py:function:: permit_directory_all(directory_path: str | pathlib.Path) -> None

   Permits a directory to be read, written and executed. Use with care as it can be a source for security issues.

   :param directory_path: Input path.
   :type directory_path: str | pathlib.Path

   :returns: None


.. py:function:: permit_script_execution(script_path: str | pathlib.Path) -> None

   Permits the execution of a script.

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


.. py:function:: write_script(directory_path: str | pathlib.Path, script: str, script_name: str) -> None

   Records a `script_name` in the `scripts` project directory.

   :param directory_path: Design directory.
   :type directory_path: str
   :param script: Script to write.
   :type script: str
   :param script_name: Name of the script.
   :type script_name: str

   :returns: None


.. py:function:: create_gdsfactory_component_from_openlane(design_name_v1: str | None = None, design_directory: str | pathlib.Path | None = None, run_name: str | None = None, v1: bool = True) -> gdsfactory.Component

   This function cretes a gdsfactory layout component that can be included in the network codesign of the device, or that can be used for interconnection codesign.

   It will look into the latest design run and extract the final OpenLane-generated GDS. You do not have to have run this with OpenLane2 as it just looks at the latest run.

   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH.
   :type design_directory: str
   :param run_name: Name of the run to extract the GDS from. If None, it will look at the latest run.
   :type run_name: str
   :param v1: If True, it will import the design from the OpenLane v1 configuration.
   :type v1: bool

   :returns: GDSFactory component.
   :rtype: component(gf.Component)


.. py:function:: get_design_directory_from_root_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> pathlib.Path

   Gets the design directory from the root directory.

   :param design_name: Name of the design.
   :type design_name: str
   :param root_directory: Design directory.
   :type root_directory: str | pathlib.Path

   :returns: Design directory.
   :rtype: design_directory(pathlib.Path)


.. py:function:: return_path(input_path: str | pathlib.Path) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems.

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: get_design_from_openlane_migration(v1: bool = True, design_name_v1: str | None = None, design_directory: str | pathlib.Path | None = None, root_directory_v1: str | pathlib.Path | None = None) -> (str, pathlib.Path)

   This function provides the integration mechanism for easily migrating the interconnection with other toolsets from an OpenLane v1 design to an OpenLane v2 design.

   This function checks if the inputs are to be treated as v1 inputs. If so, and a `design_name` is provided then it will set the `design_directory` to the corresponding `design_name` directory in the corresponding `root_directory_v1 / designs`. If no `root_directory` is provided then it returns `$OPENLANE_ROOT/"<latest>"/. If a `design_directory` is provided then this will always take precedence even with a `v1` flag.

   :param v1: If True, it will migrate from v1 to v2.
   :type v1: bool
   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH. Optional path for v2-based designs.
   :type design_directory: str
   :param root_directory_v1: Root directory of OpenLane v1. If set to None it will return `$OPENLANE_ROOT/"<latest>"
   :type root_directory_v1: str

   :returns: None


.. py:function:: find_design_run(design_directory: str | pathlib.Path, run_name: str | None = None) -> pathlib.Path

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

   They get sorted based on a reverse `list.sort()` method.


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


.. py:function:: write_configuration_openlane_v1(configuration: dict, design_directory: str | pathlib.Path) -> None

   Writes a `config.json` onto a `design_directory`

   :param configuration: OpenLane configuration dictionary.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None


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


.. py:function:: contains_in_lines(file_lines_data: pandas.DataFrame, keyword: str)

   Check if the keyword is contained in the file lines

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame
   :param keyword: Keyword to search for
   :type keyword: str

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: read_file_lines(file_path: str | pathlib.Path)

   Extract lines from the file

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

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


.. py:function:: create_file_lines_dataframe(file_lines_raw)

   Create a DataFrame from the raw lines of a file

   :param file_lines_raw: list containing the file lines
   :type file_lines_raw: list

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


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


.. py:function:: calculate_propagation_delay_from_file(file_path: str | pathlib.Path)

   Calculate the propagation delay for each frame in the file

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

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


.. py:function:: check_path_exists(path: str | pathlib.Path, raise_errors: bool = False) -> bool

   Checks if a directory exists.

   :param path: Input path.
   :type path: str | pathlib.Path

   :returns: True if directory exists.
   :rtype: directory_exists(bool)


.. py:function:: read_file(file_path: str | pathlib.Path)

   Read the file from the given path

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

   :returns: the opened file
   :rtype: file


.. py:function:: run_openlane_flow(configuration: dict | None = test_spm_open_lane_configuration, design_directory: str = '/foss/designs/spm') -> None

   Runs the OpenLane flow.

   :param configuration: OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None


.. py:function:: single_parameter_sweep(base_design_configuration: dict, parameter_name: str, parameter_sweep_values: list)

   This function takes a base_design_configuration dictionary and sweeps a single parameter over a list of values. It returns a list of dictionaries that correspond to the parameter sweep.

   :param base_design_configuration: Base design configuration dictionary.
   :type base_design_configuration: dict
   :param parameter_name: Name of parameter to sweep.
   :type parameter_name: str
   :param parameter_sweep_values: List of values to sweep.
   :type parameter_sweep_values: list

   :returns: List of dictionaries that correspond to the parameter sweep.
   :rtype: parameter_sweep_design_dictionary_array(list)


.. py:function:: multi_parameter_sweep(base_design_configuration: dict, parameter_sweep_dictionary: dict) -> list

   This multiparameter sweep is pretty cool, as it will generate designer list of dictionaries that comprise of all the possible combinations of your parameter sweeps. For example, if you are sweeping `parameter_1 = np.arange(0, 2) = array([0, 1])`, and `parameter_2 = np.arange(2, 4) = array([2, 3])`, then this function will generate list of dictionaries based on the default_design dictionary, but that will comprise of all the potential parameter combinations within this list.

   For the example above, there arould be 4 combinations [(0, 2), (0, 3), (1, 2), (1, 3)].

   If you were instead sweeping for `parameter_1 = np.arange(0, 5)` and `parameter_2 = np.arange(0, 5)`, the dictionary generated would correspond to these parameter combinations of::
       [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)].

   Make sure to use the parameter_names from default_design when writing up the parameter_sweep dictionary key name.

   Example project_structure formats::

       example_parameter_sweep_dictionary = {
           "parameter_1": np.arange(1, -40, 1),
           "parameter_2": np.arange(1, -40, 1),
       }

       example_base_design_configuration = {
           "parameter_1": 10.0,
           "parameter_2": 40.0,
           "parameter_3": 0,
       }

   :param base_design_configuration: Dictionary of the default design configuration.
   :type base_design_configuration: dict
   :param parameter_sweep_dictionary: Dictionary of the parameter sweep. The keys should be the same as the keys in the base_design_configuration dictionary.
   :type parameter_sweep_dictionary: dict

   :returns: List of dictionaries that comprise of all the possible combinations of your parameter sweeps.
   :rtype: parameter_sweep_design_dictionary_array(list)


.. py:data:: __author__
   :value: 'Dario Quintero'



.. py:data:: __email__
   :value: 'darioaquintero@gmail.com'



.. py:data:: __version__
   :value: '0.0.33'
