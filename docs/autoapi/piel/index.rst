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

   openlane/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   cli/index.rst
   cocotb/index.rst
   defaults/index.rst
   file_system/index.rst
   gdsfactory/index.rst
   parametric/index.rst
   piel/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.check_example_design
   piel.copy_source_folder
   piel.permit_script_execution
   piel.setup_example_design
   piel.return_path
   piel.run_script
   piel.write_script
   piel.configure_parametric_designs
   piel.create_parametric_designs
   piel.find_design_run
   piel.check_config_json_exists_openlane_v1
   piel.check_design_exists_openlane_v1
   piel.configure_and_run_design_openlane_v1
   piel.configure_flow_script_openlane_v1
   piel.get_latest_version_root_openlane_v1
   piel.write_configuration_openlane_v1
   piel.run_openlane_flow
   piel.single_parameter_sweep
   piel.multi_parameter_sweep
   piel.check_cocotb_testbench_exists
   piel.configure_cocotb_simulation
   piel.run_cocotb_simulation



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.test_spm_open_lane_configuration
   piel.example_open_lane_configuration
   piel.make_cocotb
   piel.write_cocotb_makefile
   piel.__author__
   piel.__email__
   piel.__version__


.. py:data:: test_spm_open_lane_configuration



.. py:data:: example_open_lane_configuration



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


.. py:function:: setup_example_design(project_source: Literal[piel, openlane] = 'piel', example_name: str = 'simple_design') -> None

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

   :param project_source: Source of the project.
   :type project_source: str
   :param example_name: Name of the example design.
   :type example_name: str

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


.. py:function:: write_script(directory_path: str | pathlib.Path, script: str, script_name: str) -> None

   Records a `script_name` in the `scripts` project directory.

   :param directory_path: Design directory.
   :type directory_path: str
   :param script: Script to write.
   :type script: str
   :param script_name: Name of the script.
   :type script_name: str

   :returns: None


.. py:function:: configure_parametric_designs(parameter_sweep_dictionary: dict, source_design_directory: str | pathlib.Path) -> list

   For a given `source_design_directory`, this function reads in the config.json file and returns a set of parametric sweeps that gets used when creating a set of parametric designs.

   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param source_design_directory: Source design directory.
   :type source_design_directory: str | pathlib.Path

   :returns: List of configurations to sweep.
   :rtype: configuration_sweep(list)


.. py:function:: create_parametric_designs(parameter_sweep_dictionary: dict, source_design_directory: str | pathlib.Path, target_directory: str | pathlib.Path) -> None

   Takes a OpenLane v1 source directory and creates a parametric combination of these designs.

   :param parameter_sweep_dictionary: Dictionary of parameters to sweep.
   :type parameter_sweep_dictionary: dict
   :param source_design_directory: Source design directory.
   :type source_design_directory: str
   :param target_directory: Target directory.
   :type target_directory: str

   :returns: None


.. py:function:: find_design_run(design_directory: str | pathlib.Path, run_name: str | None = None) -> str

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


.. py:function:: configure_flow_script_openlane_v1(design_name: str, root_directory: str | pathlib.Path | None = None) -> None

   Configures the OpenLane v1 flow script after checking that the design directory exists.

   :param design_directory: Design directory. Defaults to latest OpenLane root.
   :type design_directory: str | pathlib.Path | None

   :returns: None


.. py:function:: get_latest_version_root_openlane_v1() -> pathlib.Path

   Gets the latest version root of OpenLane v1.


.. py:function:: write_configuration_openlane_v1(configuration: dict, design_directory: str | pathlib.Path) -> None

   Writes a `config.json` onto a `design_directory`

   :param configuration: OpenLane configuration dictionary.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None


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


.. py:function:: check_cocotb_testbench_exists(design_directory: str | pathlib.Path) -> bool

   Checks if a cocotb testbench exists in the design directory.

   :param design_directory: Design directory.
   :type design_directory: str | pathlib.Path

   :returns: True if cocotb testbench exists.
   :rtype: cocotb_testbench_exists(bool)


.. py:function:: configure_cocotb_simulation(design_directory: str | pathlib.Path, simulator: Literal[icarus, verilator], top_level_language: Literal[verilog, vhdl], top_level_verilog_module: str, test_python_module: str, design_sources_list: list | None = None)

   Writes a cocotb makefile.

   If no design_sources_list is provided then it adds all the design sources under the `src` folder.

   In the form::
       Makefile
       # defaults
       SIM ?= icarus
       TOPLEVEL_LANG ?= verilog

       VERILOG_SOURCES += $(PWD)/my_design.sv
       # use VHDL_SOURCES for VHDL files

       # TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
       TOPLEVEL = my_design

       # MODULE is the basename of the Python test file
       MODULE = test_my_design

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


.. py:data:: make_cocotb



.. py:function:: run_cocotb_simulation(design_directory: str) -> subprocess.CompletedProcess

   Equivalent to running the cocotb makefile::
       make

   :param design_directory: The directory where the design is located.
   :type design_directory: str

   :returns: The subprocess.CompletedProcess object.
   :rtype: subprocess.CompletedProcess


.. py:data:: write_cocotb_makefile



.. py:data:: __author__
   :value: 'Dario Quintero'



.. py:data:: __email__
   :value: 'darioaquintero@gmail.com'



.. py:data:: __version__
   :value: '0.0.27'
