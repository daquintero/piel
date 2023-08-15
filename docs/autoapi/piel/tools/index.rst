:py:mod:`piel.tools`
====================

.. py:module:: piel.tools


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   cocotb/index.rst
   gdsfactory/index.rst
   hdl21/index.rst
   openlane/index.rst
   qutip/index.rst
   sax/index.rst
   thewalrus/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.check_cocotb_testbench_exists
   piel.tools.configure_cocotb_simulation
   piel.tools.run_cocotb_simulation
   piel.tools.get_simulation_output_files_from_design
   piel.tools.read_simulation_data
   piel.tools.simple_plot_simulation_data
   piel.tools.get_input_ports_index
   piel.tools.get_matched_ports_tuple_index
   piel.tools.get_design_from_openlane_migration
   piel.tools.find_design_run
   piel.tools.check_config_json_exists_openlane_v1
   piel.tools.check_design_exists_openlane_v1
   piel.tools.configure_and_run_design_openlane_v1
   piel.tools.configure_parametric_designs_openlane_v1
   piel.tools.configure_flow_script_openlane_v1
   piel.tools.create_parametric_designs_openlane_v1
   piel.tools.get_design_directory_from_root_openlane_v1
   piel.tools.get_latest_version_root_openlane_v1
   piel.tools.read_configuration_openlane_v1
   piel.tools.write_configuration_openlane_v1
   piel.tools.filter_timing_sta_files
   piel.tools.filter_power_sta_files
   piel.tools.get_all_timing_sta_files
   piel.tools.get_all_power_sta_files
   piel.tools.calculate_max_frame_amount
   piel.tools.calculate_propagation_delay_from_file
   piel.tools.calculate_propagation_delay_from_timing_data
   piel.tools.configure_timing_data_rows
   piel.tools.configure_frame_id
   piel.tools.filter_timing_data_by_net_name_and_type
   piel.tools.get_frame_meta_data
   piel.tools.get_frame_lines_data
   piel.tools.get_frame_timing_data
   piel.tools.get_all_timing_data_from_file
   piel.tools.read_sta_rpt_fwf_file
   piel.tools.contains_in_lines
   piel.tools.create_file_lines_dataframe
   piel.tools.get_file_line_by_keyword
   piel.tools.read_file_lines
   piel.tools.run_openlane_flow
   piel.tools.configure_ngspice_simulation
   piel.tools.configure_operating_point_simulation
   piel.tools.configure_transient_simulation
   piel.tools.run_simulation
   piel.tools.convert_numeric_to_prefix
   piel.tools.address_value_dictionary_to_function_parameter_dictionary
   piel.tools.compose_recursive_instance_location
   piel.tools.get_component_instances
   piel.tools.get_netlist_instances_by_prefix
   piel.tools.get_matched_model_recursive_netlist_instances
   piel.tools.get_sdense_ports_index
   piel.tools.sax_to_s_parameters_standard_matrix
   piel.tools.all_fock_states_from_photon_number
   piel.tools.convert_qobj_to_jax
   piel.tools.convert_output_type
   piel.tools.fock_state_nonzero_indexes
   piel.tools.fock_state_to_photon_number_factorial
   piel.tools.fock_states_at_mode_index
   piel.tools.verify_matrix_is_unitary
   piel.tools.subunitary_selection_on_range
   piel.tools.subunitary_selection_on_index



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.tools.delete_simulation_output_files
   piel.tools.get_simulation_output_files
   piel.tools.snet
   piel.tools.standard_s_parameters_to_qutip_qobj


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


.. py:data:: get_simulation_output_files



.. py:function:: get_simulation_output_files_from_design(design_directory: piel.config.piel_path_types, extension: str = 'csv')

   This function returns a list of all the simulation output files in the design directory.

   :param design_directory: The path to the design directory.
   :type design_directory: piel_path_types

   :returns: List of all the simulation output files in the design directory.
   :rtype: output_files (list)


.. py:function:: read_simulation_data(file_path: piel.config.piel_path_types)

   This function returns a Pandas dataframe that contains all the simulation data outputted from the simulation run.

   :param file_path: The path to the simulation data file.
   :type file_path: piel_path_types

   :returns: The simulation data in a Pandas dataframe.
   :rtype: simulation_data (pd.DataFrame)


.. py:function:: simple_plot_simulation_data(simulation_data: pandas.DataFrame)


.. py:function:: get_input_ports_index(ports_index: dict, sorting_algorithm: Literal[get_input_ports_index.prefix] = 'prefix', prefix: str = 'in') -> tuple

   This function returns the input ports of a component. However, input ports may have different sets of prefixes and suffixes. This function implements different sorting algorithms for different ports names. The default algorithm is `prefix`, which sorts the ports by their prefix. The Endianness implementation means that the tuple order is determined according to the last numerical index order of the port numbering.

   .. code-block:: python

       raw_ports_index = {
           "in_o_0": 0,
           "out_o_0": 1,
           "out_o_1": 2,
           "out_o_2": 3,
           "out_o_3": 4,
           "in_o_1": 5,
           "in_o_2": 6,
           "in_o_3": 7,
       }

       get_input_ports_index(ports_index=raw_ports_index)

       # Output
       ((0, "in_o_0"), (5, "in_o_1"), (6, "in_o_2"), (7, "in_o_3"))

   :param ports_index: The ports index dictionary.
   :type ports_index: dict
   :param sorting_algorithm: The sorting algorithm to use. Defaults to "prefix".
   :type sorting_algorithm: Literal["prefix"], optional
   :param prefix: The prefix to use for the sorting algorithm. Defaults to "in".
   :type prefix: str, optional

   :returns: The ordered input ports index tuple.
   :rtype: tuple


.. py:function:: get_matched_ports_tuple_index(ports_index: dict, selected_ports_tuple: Optional[tuple] = None, sorting_algorithm: Literal[get_matched_ports_tuple_index.prefix, selected_ports] = 'prefix', prefix: str = 'in') -> (tuple, tuple)

   This function returns the input ports of a component. However, input ports may have different sets of prefixes
   and suffixes. This function implements different sorting algorithms for different ports names. The default
   algorithm is `prefix`, which sorts the ports by their prefix. The Endianness implementation means that the tuple
   order is determined according to the last numerical index order of the port numbering. Returns just a tuple of
   the index.

   .. code-block:: python

       raw_ports_index = {
           "in_o_0": 0,
           "out_o_0": 1,
           "out_o_1": 2,
           "out_o_2": 3,
           "out_o_3": 4,
           "in_o_1": 5,
           "in_o_2": 6,
           "in_o_3": 7,
       }

       get_input_ports_tuple_index(ports_index=raw_ports_index)

       # Output
       (0, 5, 6, 7)

   :param ports_index: The ports index dictionary.
   :type ports_index: dict
   :param selected_ports_tuple: The selected ports tuple. Defaults to None.
   :type selected_ports_tuple: tuple, optional
   :param sorting_algorithm: The sorting algorithm to use. Defaults to "prefix".
   :type sorting_algorithm: Literal["prefix"], optional
   :param prefix: The prefix to use for the sorting algorithm. Defaults to "in".
   :type prefix: str, optional

   :returns: The ordered input ports index tuple.
             matched_ports_name_tuple_order(tuple): The ordered input ports name tuple.
   :rtype: matches_ports_index_tuple_order(tuple)


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


.. py:function:: find_design_run(design_directory: piel.config.piel_path_types, run_name: str | None = None) -> pathlib.Path

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

   They get sorted based on a reverse `list.sort()` method.

   # TODO docs


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


.. py:function:: write_configuration_openlane_v1(configuration: dict, design_directory: str | pathlib.Path) -> None

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


.. py:function:: run_openlane_flow(configuration: dict | None = test_spm_open_lane_configuration, design_directory: piel.config.piel_path_types = '/foss/designs/spm') -> None

   Runs the OpenLane flow.

   :param configuration: OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: piel_path_types

   :returns: None


.. py:function:: configure_ngspice_simulation(run_directory: piel.config.piel_path_types = '.')

   This function configures the NGSPICE simulation for the circuit and returns a simulation class.

   :param run_directory: Directory where the simulation will be run
   :type run_directory: piel_path_types

   :returns: Configured NGSPICE simulation options
   :rtype: simulation_options


.. py:function:: configure_operating_point_simulation(testbench: hdl21.Module, **kwargs)

   This function configures the DC operating point simulation for the circuit and returns a simulation class.

   :param testbench: HDL21 testbench
   :type testbench: Module
   :param \*\*kwargs: Additional arguments to be passed to the operating point simulation such as name.

   :returns: HDL21 simulation class
   :rtype: Simulation


.. py:function:: configure_transient_simulation(testbench: hdl21.Module, stop_time_s: float, step_time_s: float, **kwargs)

   This function configures the transient simulation for the circuit and returns a simulation class.

   :param testbench: HDL21 testbench
   :type testbench: Module
   :param stop_time_s: Stop time of the simulation in seconds
   :type stop_time_s: float
   :param step_time_s: Step time of the simulation in seconds
   :type step_time_s: float
   :param \*\*kwargs: Additional arguments to be passed to the transient simulation

   :returns: HDL21 simulation class
   :rtype: Simulation


.. py:function:: run_simulation(simulation: hdl21.sim.Sim, simulator_name: Literal[ngspice] = 'ngspice', simulation_options: Optional[vlsirtools.spice.SimOptions] = None, to_csv: bool = True)

   This function runs the transient simulation for the circuit and returns the results.

   :param simulation: HDL21 simulation class
   :type simulation: h.sim.Sim
   :param simulator_name: Name of the simulator
   :type simulator_name: Literal["ngspice"]
   :param simulation_options: Simulation options
   :type simulation_options: Optional[vsp.SimOptions]
   :param to_csv: Whether to save the results to a csv file
   :type to_csv: bool

   :returns: Simulation results
   :rtype: results


.. py:function:: convert_numeric_to_prefix(value: float)

   This function converts a numeric value to a number under a SPICE unit closest to the base prefix. This allows us to connect a particular number real output, into a term that can be used in a SPICE netlist.


.. py:function:: address_value_dictionary_to_function_parameter_dictionary(address_value_dictionary: dict, parameter_key: str)

   This function converts an address of an instance with particular parameter values in the form:

       {('component_lattice_gener_fb8c4da8', 'mzi_1', 'sxt'): 0,
       ('component_lattice_gener_fb8c4da8', 'mzi_5', 'sxt'): 0}

   to

       {'mzi_1': {'sxt': {parameter_key: 0}},
       ('mzi_5', {'sxt': {parameter_key: 0}}}




.. py:function:: compose_recursive_instance_location(recursive_netlist: dict, top_level_instance_name: str, required_models: list, target_component_prefix: str, models: dict)

      This function returns the recursive location of any matching ``target_component_prefix`` instances within the ``recursive_netlist``. A function that returns the mapping of the ``matched_component`` in the corresponding netlist at any particular level of recursion. This function iterates over a particular level of recursion of a netlist. It returns a list of the missing required components, and updates a dictionary of models that contains a particular matching component. It returns the corresponding list of instances of a particular component at that level of recursion, so that it can be appended upon in order to construct the location of the corresponding matching elements.

      If ``required_models`` is an empty list, it means no recursion is required and the function is complete. If a ``required_model_i`` in ``required_models`` matches ``target_component_prefix``, then no more recursion is required down the component function.

      The ``recursive_netlist`` should contain all the missing composed models that are not provided in the main models dictionary. If not, then we need to require the user to input the missing model that cannot be extracted from the composed netlist.
   We know when a model is composed, and when it is already provided at every level of recursion based on the ``models`` dictionary that gets updated at each level of recursion with the corresponding models of that level, and the ``required_models`` down itself.

      However, a main question appears on how to do the recursion. There needs to be a flag that determines that the recursion is complete. However, this is only valid for every particular component in the ``required_models`` list. Every component might have missing component. This means that this recursion begins component by component, updating the ``required_models`` list until all of them have been composed from the recursion or it is determined that is it missing fully.

      It would be ideal to access the particular component that needs to be implemented.

      Returns a tuple of ``model_composition_mapping, instance_composition_mapping, target_component_mapping`` in the form of

          ({'mzi_214beef3': ['straight_heater_metal_s_ad3c1693']},
           {'mzi_214beef3': ['mzi_1', 'mzi_5'],
            'mzi_d46c281f': ['mzi_2', 'mzi_3', 'mzi_4']})


.. py:function:: get_component_instances(recursive_netlist: dict, top_level_prefix: str, component_name_prefix: str)

   Returns a dictionary of all instances of a given component in a recursive netlist.

   :param recursive_netlist: The recursive netlist to search.
   :param top_level_prefix: The prefix of the top level instance.
   :param component_name_prefix: The name of the component to search for.

   :returns: A dictionary of all instances of the given component.


.. py:function:: get_netlist_instances_by_prefix(recursive_netlist: dict, instance_prefix: str) -> str

   Returns a list of all instances with a given prefix in a recursive netlist.

   :param recursive_netlist: The recursive netlist to search.
   :param instance_prefix: The prefix to search for.

   :returns: A list of all instances with the given prefix.


.. py:function:: get_matched_model_recursive_netlist_instances(recursive_netlist: dict, top_level_instance_prefix: str, target_component_prefix: str, models: Optional[dict] = None) -> list[tuple]

   This function returns an active component list with a tuple mapping of the location of the active component within the recursive netlist and corresponding model. It will recursively look within a netlist to locate what models use a particular component model. At each stage of recursion, it will compose a list of the elements that implement this matching model in order to relate the model to the instance, and hence the netlist address of the component that needs to be updated in order to functionally implement the model.

   It takes in as a set of parameters the recursive_netlist generated by a ``gdsfactory`` netlist implementation.

   Returns a list of tuples, that correspond to the phases applied with the corresponding component paths at multiple levels of recursion.
   eg. [("component_lattice_gener_fb8c4da8", "mzi_1", "sxt"), ("component_lattice_gener_fb8c4da8", "mzi_5", "sxt")] and these are our keys to our sax circuit decomposition.


.. py:function:: get_sdense_ports_index(input_ports_order: tuple, all_ports_index: dict) -> dict

   This function returns the ports index of the sax dense S-parameter matrix.

   Given that the order of the iteration is provided by the user, the dictionary keys will also be ordered
   accordingly when iterating over them. This requires the user to provide a set of ordered.

   TODO verify reasonable iteration order.

   .. code-block:: python

       # The input_ports_order can be a tuple of tuples that contain the index and port name. Eg.
       input_ports_order = ((0, "in_o_0"), (5, "in_o_1"), (6, "in_o_2"), (7, "in_o_3"))
       # The all_ports_index is a dictionary of the ports index. Eg.
       all_ports_index = {
           "in_o_0": 0,
           "out_o_0": 1,
           "out_o_1": 2,
           "out_o_2": 3,
           "out_o_3": 4,
           "in_o_1": 5,
           "in_o_2": 6,
           "in_o_3": 7,
       }
       # Output
       {"in_o_0": 0, "in_o_1": 5, "in_o_2": 6, "in_o_3": 7}

   :param input_ports_order: The ports order tuple. Can be a tuple of tuples that contain the index and port name.
   :type input_ports_order: tuple
   :param all_ports_index: The ports index dictionary.
   :type all_ports_index: dict

   :returns: The ordered input ports index tuple.
   :rtype: tuple


.. py:function:: sax_to_s_parameters_standard_matrix(sax_input: sax.SType, input_ports_order: tuple | None = None) -> tuple

   A ``sax`` S-parameter SDict is provided as a dictionary of tuples with (port0, port1) as the key. This
   determines the direction of the scattering relationship. It means that the number of terms in an S-parameter
   matrix is the number of ports squared.

   In order to generalise, this function returns both the S-parameter matrices and the indexing ports based on the
   amount provided. In terms of computational speed, we definitely would like this function to be algorithmically
   very fast. For now, I will write a simple python implementation and optimise in the future.

   It is possible to see the `sax` SDense notation equivalence here:
   https://flaport.github.io/sax/nbs/08_backends.html

   .. code-block:: python

       import jax.numpy as jnp
       from sax.core import SDense

       # Directional coupler SDense representation
       dc_sdense: SDense = (
           jnp.array([[0, 0, τ, κ], [0, 0, κ, τ], [τ, κ, 0, 0], [κ, τ, 0, 0]]),
           {"in0": 0, "in1": 1, "out0": 2, "out1": 3},
       )


       # Directional coupler SDict representation
       # Taken from https://flaport.github.io/sax/nbs/05_models.html
       def coupler(*, coupling: float = 0.5) -> SDict:
           kappa = coupling**0.5
           tau = (1 - coupling) ** 0.5
           sdict = reciprocal(
               {
                   ("in0", "out0"): tau,
                   ("in0", "out1"): 1j * kappa,
                   ("in1", "out0"): 1j * kappa,
                   ("in1", "out1"): tau,
               }
           )
           return sdict

   If we were to relate the mapping accordingly based on the ports indexes, a S-Parameter matrix in the form of
   :math:`S_{(output,i),(input,i)}` would be:

   .. math::

       S = \begin{bmatrix}
               S_{00} & S_{10} \\
               S_{01} & S_{11} \\
           \end{bmatrix} =
           \begin{bmatrix}
           \tau & j \kappa \\
           j \kappa & \tau \\
           \end{bmatrix}

   Note that the standard S-parameter and hence unitary representation is in the form of:

   .. math::

       S = \begin{bmatrix}
               S_{00} & S_{01} \\
               S_{10} & S_{11} \\
           \end{bmatrix}


   .. math::

       \begin{bmatrix}
           b_{1} \\
           \vdots \\
           b_{n}
       \end{bmatrix}
       =
       \begin{bmatrix}
           S_{11} & \dots & S_{1n} \\
           \vdots & \ddots & \vdots \\
           S_{n1} & \dots & S_{nn}
       \end{bmatrix}
       \begin{bmatrix}
           a_{1} \\
           \vdots \\
           a_{n}
       \end{bmatrix}

   TODO check with Floris, does this mean we need to transpose the matrix?

   :param sax_input: The sax S-parameter dictionary.
   :type sax_input: sax.SType
   :param input_ports_order: The ports order tuple containing the names and order of the input ports.
   :type input_ports_order: tuple

   :returns: The S-parameter matrix and the input ports index tuple in the standard S-parameter notation.
   :rtype: tuple


.. py:data:: snet



.. py:function:: all_fock_states_from_photon_number(mode_amount: int, photon_amount: int = 1, output_type: Literal[qutip, jax] = 'qutip') -> list

   For a specific amount of modes, we can generate all the possible Fock states for whatever amount of input photons we desire. This returns a list of all corresponding Fock states.

   :param mode_amount: The amount of modes in the system.
   :type mode_amount: int
   :param photon_amount: The amount of photons in the system. Defaults to 1.
   :type photon_amount: int, optional
   :param output_type: The type of output. Defaults to "qutip".
   :type output_type: str, optional

   :returns: A list of all the Fock states.
   :rtype: list


.. py:function:: convert_qobj_to_jax(qobj: qutip.Qobj) -> jax.numpy.ndarray


.. py:function:: convert_output_type(array: numpy.ndarray, output_type: Literal[qutip, jax])


.. py:function:: fock_state_nonzero_indexes(fock_state: qutip.Qobj | jax.numpy.ndarray) -> tuple[int]

   This function returns the indexes of the nonzero elements of a Fock state.

   :param fock_state: A QuTip QObj representation of the Fock state.
   :type fock_state: qutip.Qobj

   :returns: The indexes of the nonzero elements of the Fock state.
   :rtype: tuple


.. py:function:: fock_state_to_photon_number_factorial(fock_state: qutip.Qobj | jax.numpy.ndarray) -> float

       This function converts a Fock state defined as:

       .. math::

   ewcommand{\ket}[1]{\left|{#1}
   ight
   angle}
           \ket{f_1} = \ket{j_1, j_2, ... j_N}$

       and returns:

       .. math::

           j_1^{'}! j_2^{'}! ... j_N^{'}!

       Args:
           fock_state (qutip.Qobj): A QuTip QObj representation of the Fock state.

       Returns:
           float: The photon number factorial of the Fock state.



.. py:function:: fock_states_at_mode_index(mode_amount: int, target_mode_index: int, maximum_photon_amount: Optional[int] = 1, output_type: Literal[qutip, jax] = 'qutip') -> list

   This function returns a list of valid Fock states that fulfill a condition of having a maximum photon number at a specific mode index.

   :param mode_amount: The amount of modes in the system.
   :type mode_amount: int
   :param target_mode_index: The mode index to check the photon number at.
   :type target_mode_index: int
   :param maximum_photon_amount: The amount of photons in the system. Defaults to 1.
   :type maximum_photon_amount: int, optional
   :param output_type: The type of output. Defaults to "qutip".
   :type output_type: str, optional

   :returns: A list of all the Fock states.
   :rtype: list


.. py:data:: standard_s_parameters_to_qutip_qobj



.. py:function:: verify_matrix_is_unitary(matrix: jax.numpy.ndarray) -> bool

   Verify that the matrix is unitary.

   :param matrix: The matrix to verify.
   :type matrix: jnp.ndarray

   :returns: True if the matrix is unitary, False otherwise.
   :rtype: bool


.. py:function:: subunitary_selection_on_range(unitary_matrix: jax.numpy.ndarray, stop_index: tuple, start_index: Optional[tuple] = (0, 0))

   This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
   the output matrix is also a unitary.

   TODO implement validation of a 2D matrix.


.. py:function:: subunitary_selection_on_index(unitary_matrix: jax.numpy.ndarray, rows_index: jax.numpy.ndarray | tuple, columns_index: jax.numpy.ndarray | tuple)

   This function returns a unitary between the indexes selected, and verifies the indexes are valid by checking that
   the output matrix is also a unitary.

   TODO implement validation of a 2D matrix.
