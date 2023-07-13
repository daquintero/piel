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

   components/index.rst
   integration/index.rst
   models/index.rst
   tools/index.rst
   visual/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   cli/index.rst
   config/index.rst
   defaults/index.rst
   file_conversion/index.rst
   file_system/index.rst
   parametric/index.rst
   project_structure/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

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
   piel.gdsfactory_netlist_to_pyspice
   piel.spice_netlist_to_pyspice_circuit
   piel.gdsfactory_netlist_to_spice_netlist
   piel.sax_to_s_parameters_standard_matrix
   piel.unitary_permanent
   piel.sax_circuit_permanent
   piel.sax_to_ideal_qutip_unitary
   piel.standard_s_parameters_to_ideal_qutip_unitary
   piel.single_parameter_sweep
   piel.multi_parameter_sweep
   piel.check_cocotb_testbench_exists
   piel.configure_cocotb_simulation
   piel.run_cocotb_simulation
   piel.get_simulation_output_files_from_design
   piel.read_simulation_data
   piel.simple_plot_simulation_data
   piel.get_input_ports_index
   piel.get_matched_ports_tuple_index
   piel.get_design_from_openlane_migration
   piel.find_design_run
   piel.check_config_json_exists_openlane_v1
   piel.check_design_exists_openlane_v1
   piel.configure_and_run_design_openlane_v1
   piel.configure_parametric_designs_openlane_v1
   piel.configure_flow_script_openlane_v1
   piel.create_parametric_designs_openlane_v1
   piel.get_design_directory_from_root_openlane_v1
   piel.get_latest_version_root_openlane_v1
   piel.read_configuration_openlane_v1
   piel.write_configuration_openlane_v1
   piel.filter_timing_sta_files
   piel.filter_power_sta_files
   piel.get_all_timing_sta_files
   piel.get_all_power_sta_files
   piel.calculate_max_frame_amount
   piel.calculate_propagation_delay_from_file
   piel.calculate_propagation_delay_from_timing_data
   piel.configure_timing_data_rows
   piel.configure_frame_id
   piel.filter_timing_data_by_net_name_and_type
   piel.get_frame_meta_data
   piel.get_frame_lines_data
   piel.get_frame_timing_data
   piel.get_all_timing_data_from_file
   piel.read_sta_rpt_fwf_file
   piel.contains_in_lines
   piel.create_file_lines_dataframe
   piel.get_file_line_by_keyword
   piel.read_file_lines
   piel.run_openlane_flow
   piel.get_sdense_ports_index
   piel.sax_to_s_parameters_standard_matrix



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.numerical_solver
   piel.nso
   piel.piel_path_types
   piel.test_spm_open_lane_configuration
   piel.example_open_lane_configuration
   piel.delete_simulation_output_files
   piel.get_simulation_output_files
   piel.snet
   piel.__author__
   piel.__email__
   piel.__version__


.. py:data:: numerical_solver



.. py:data:: nso



.. py:data:: piel_path_types



.. py:data:: test_spm_open_lane_configuration



.. py:data:: example_open_lane_configuration



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


.. py:function:: delete_path_list_in_directory(directory_path: piel.config.piel_path_types, path_list: list, ignore_confirmation: bool = False, validate_individual: bool = False) -> None

   Deletes a list of files in a directory.

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

   :param path: Input path.
   :type path: piel_path_types
   :param extension: File extension.
   :type extension: str

   :returns: List of files.
   :rtype: file_list(list)


.. py:function:: permit_directory_all(directory_path: piel.config.piel_path_types) -> None

   Permits a directory to be read, written and executed. Use with care as it can be a source for security issues.

   :param directory_path: Input path.
   :type directory_path: piel_path_types

   :returns: None


.. py:function:: permit_script_execution(script_path: piel.config.piel_path_types) -> None

   Permits the execution of a script.

   :param script_path: Script path.
   :type script_path: piel_path_types

   :returns: None


.. py:function:: setup_example_design(project_source: Literal[piel, openlane] = 'piel', example_name: str = 'simple_design') -> None

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.

   :param project_source: Source of the project.
   :type project_source: str
   :param example_name: Name of the example design.
   :type example_name: str

   :returns: None


.. py:function:: read_json(path: piel.config.piel_path_types) -> dict

   Reads a JSON file.

   :param path: Input path.
   :type path: piel_path_types

   :returns: JSON data.
   :rtype: json_data(dict)


.. py:function:: return_path(input_path: piel.config.piel_path_types) -> pathlib.Path

   Returns a pathlib.Path to be able to perform operations accordingly internally.

   This allows us to maintain compatibility between POSIX and Windows systems.

   :param input_path: Input path.
   :type input_path: str

   :returns: Pathlib path.
   :rtype: pathlib.Path


.. py:function:: run_script(script_path: piel.config.piel_path_types) -> None

   Runs a script on the filesystem `script_path`.

   :param script_path: Script path.
   :type script_path: piel_path_types

   :returns: None


.. py:function:: write_script(directory_path: piel.config.piel_path_types, script: str, script_name: str) -> None

   Records a `script_name` in the `scripts` project directory.

   :param directory_path: Design directory.
   :type directory_path: piel_path_types
   :param script: Script to write.
   :type script: str
   :param script_name: Name of the script.
   :type script_name: str

   :returns: None


.. py:function:: create_gdsfactory_component_from_openlane(design_name_v1: str | None = None, design_directory: piel.config.piel_path_types | None = None, run_name: str | None = None, v1: bool = True) -> gdsfactory.Component

   This function cretes a gdsfactory layout component that can be included in the network codesign of the device, or that can be used for interconnection codesign.

   It will look into the latest design run and extract the final OpenLane-generated GDS. You do not have to have run this with OpenLane2 as it just looks at the latest run.

   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH.
   :type design_directory: piel_path_types
   :param run_name: Name of the run to extract the GDS from. If None, it will look at the latest run.
   :type run_name: str
   :param v1: If True, it will import the design from the OpenLane v1 configuration.
   :type v1: bool

   :returns: GDSFactory component.
   :rtype: component(gf.Component)


.. py:function:: gdsfactory_netlist_to_pyspice(gdsfactory_netlist: dict, return_raw_spice: bool = False)

   This function converts a GDSFactory electrical netlist into a standard PySpice configuration. It follows the same
   principle as the `sax` circuit composition. It returns a PySpice circuit and can return it in raw_spice form if
   necessary.

   Each GDSFactory netlist has a set of instances, each with a corresponding model, and each instance with a given
   set of geometrical settings that can be applied to each particular model. We know the type of SPICE model from
   the instance model we provides.

   We know that the gdsfactory has a set of instances, and we can map unique models via sax through our own
   composition circuit. Write the SPICE component based on the model into a total circuit representation in string
   from the reshaped gdsfactory dictionary into our own structure.


.. py:function:: spice_netlist_to_pyspice_circuit(spice_netlist: dict)

   This function converts a SPICE netlist into a PySpice circuit.

   # TODO implement validators


.. py:function:: gdsfactory_netlist_to_spice_netlist(gdsfactory_netlist: dict, models=None)

   This function maps the connections of a netlist to a node that can be used in a SPICE netlist. SPICE netlists are
   in the form of:

   .. code-block::

       RXXXXXXX N1 N2 <VALUE> <MNAME> <L=LENGTH> <W=WIDTH> <TEMP=T>

   This means that every instance, is an electrical type, and we define the two particular nodes in which it is
   connected. This means we need to convert the gdsfactory dictionary netlist into a form that allows us to map the
   connectivity for every instance. Then we can define that as a line of the SPICE netlist with a particular
   electrical model. For passives this works fine when it's a two port network such as sources, or electrical
   elements. However, non-passive elements like transistors have three ports or more which are provided in an ordered form.

   This means that the order of translations is as follows:

   .. code-block::

       1. Extract all instances and required models from the netlist
       2. Verify that the models have been provided. Each model describes the type of component this is, how many ports it requires and so on.
       3. Map the connections to each instance port as part of the instance dictionary.

   We should get as an output a dictionary in the structure:

   .. code-block::

       {
           instance_1: {
               ...
               "connections": [('straight_1,e1', 'taper_1,e2'),
                               ('straight_1,e2', 'taper_2,e2')],
               'spice_nets': {'e1': 'straight_1__e1___taper_1__e2',
                       'e2': 'straight_1__e2___taper_2__e2'},
               'spice_model': <function piel.models.physical.electronic.spice.resistor.basic_resistor()>},
           }
           ...
       }


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


.. py:function:: unitary_permanent(unitary_matrix: numpy.ndarray) -> tuple

   The permanent of a unitary is used to determine the state probability of combinatorial Gaussian boson samping systems.

   ``thewalrus`` Ryser's algorithm permananet implementation is described here: https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

   # TODO maybe implement subroutine if computation is taking forever.
   # TODO why two outputs? Understand this properly later.

   :param unitary_permanent: The unitary matrix.
   :type unitary_permanent: np.ndarray

   :returns: The circuit permanent and the time it took to compute it.
   :rtype: tuple


.. py:function:: sax_circuit_permanent(sax_input: sax.SType) -> tuple

   The permanent of a unitary is used to determine the state probability of combinatorial Gaussian boson samping systems.

   ``thewalrus`` Ryser's algorithm permananet implementation is described here: https://the-walrus.readthedocs.io/en/latest/gallery/permanent_tutorial.html

   # TODO maybe implement subroutine if computation is taking forever.

   :param sax_input: The sax S-parameter dictionary.
   :type sax_input: sax.SType

   :returns: The circuit permanent and the time it took to compute it.
   :rtype: tuple


.. py:function:: sax_to_ideal_qutip_unitary(sax_input: sax.SType)

   This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
   dimensions of the matrix can be observed.

   I think this means we need to transpose the output of the filtered sax SDense matrix to map it to a QuTip matrix.
   Note that the documentation and formatting of the standard `sax` mapping to a S-parameter standard notation is
   already in described in piel/piel/sax/utils.py.

   From this stage we can implement a ``QObj`` matrix accordingly and perform simulations accordingly.
   https://qutip.org/docs/latest/guide/qip/qip-basics.html#unitaries

   For example, a ``qutip`` representation of an s-gate gate would be:

   ..code-block:: python

       import numpy as np
       import qutip
       # S-Gate
       s_gate_matrix = np.array([[1.,   0], [0., 1.j]])
       s_gate = qutip.Qobj(mat, dims=[[2], [2]])

   In mathematical notation, this S-gate would be written as:

   ..math::

       S = \begin{bmatrix}
           1 & 0 \
           0 & i \
       \end{bmatrix}

   :param sax_input: A dictionary of S-parameters in the form of a SDict from `sax`.
   :type sax_input: sax.SType

   :returns: A QuTip QObj representation of the S-parameters in a unitary matrix.
   :rtype: qobj_unitary (qutip.Qobj)


.. py:function:: standard_s_parameters_to_ideal_qutip_unitary(s_parameters_standard_matrix: piel.config.nso.ndarray)

   This function converts the calculated S-parameters into a standard Unitary matrix topology so that the shape and
   dimensions of the matrix can be observed.

   I think this means we need to transpose the output of the filtered sax SDense matrix to map it to a QuTip matrix.
   Note that the documentation and formatting of the standard `sax` mapping to a S-parameter standard notation is
   already in described in piel/piel/sax/utils.py.

   From this stage we can implement a ``QObj`` matrix accordingly and perform simulations accordingly. https://qutip.org/docs/latest/guide/qip/qip-basics.html#unitaries

   For example, a ``qutip`` representation of an s-gate gate would be:

   ..code-block:: python

       import numpy as np
       import qutip

       # S-Gate
       s_gate_matrix = np.array([[1.,   0], [0., 1.j]])
       s_gate = qutip.Qobj(mat, dims=[[2], [2]])

   In mathematical notation, this S-gate would be written as:

   ..math::

       S = \begin{bmatrix}
           1 & 0 \
           0 & i \
       \end{bmatrix}

   :param s_parameters_standard_matrix: A dictionary of S-parameters in the form of a SDict from `sax`.
   :type s_parameters_standard_matrix: nso.ndarray

   :returns: A QuTip QObj representation of the S-parameters in a unitary matrix.
   :rtype: qobj_unitary (qutip.Qobj)


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



.. py:data:: __author__
   :value: 'Dario Quintero'



.. py:data:: __email__
   :value: 'darioaquintero@gmail.com'



.. py:data:: __version__
   :value: '0.0.42'
