:py:mod:`piel`
==============

.. py:module:: piel

.. autoapi-nested-parse::

   Top-level package for piel.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   cli/index.rst
   defaults/index.rst
   file_system/index.rst
   gdsfactory/index.rst
   openlane_v1/index.rst
   openlane_v2/index.rst
   parametric/index.rst
   piel/index.rst
   simulation/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.copy_source_folder
   piel.setup_example_design
   piel.check_example_design
   piel.write_openlane_configuration
   piel.run_openlane_flow
   piel.single_parameter_sweep
   piel.multi_parameter_sweep
   piel.configure_cocotb_simulation
   piel.run_cocotb_simulation



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.__author__
   piel.__email__
   piel.__version__
   piel.test_spm_open_lane_configuration
   piel.example_open_lane_configuration
   piel.make_cocotb
   piel.write_cocotb_makefile


.. py:data:: __author__
   :value: 'Dario Quintero'

   

.. py:data:: __email__
   :value: 'darioaquintero@gmail.com'

   

.. py:data:: __version__
   :value: '0.0.25'

   

.. py:data:: test_spm_open_lane_configuration

   

.. py:data:: example_open_lane_configuration

   

.. py:function:: copy_source_folder(source_directory: str, target_directory: str)


.. py:function:: setup_example_design(project_source: Literal[piel, openlane] = 'piel', example_name: str = 'simple_design')

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.


.. py:function:: check_example_design(example_name: str = 'simple_design')

   We copy the example simple_design from docs to the `/foss/designs` in the `iic-osic-tools` environment.


.. py:function:: write_openlane_configuration(project_directory=None, configuration=dict())


.. py:function:: run_openlane_flow(configuration: dict | None = test_spm_open_lane_configuration, design_directory: str = '/foss/designs/spm') -> None

   Runs the OpenLane flow.

   :param configuration: OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: str

   :returns: None


.. py:function:: single_parameter_sweep(base_design_configuration: dict, parameter_name: str, parameter_sweep_values: list)


.. py:function:: multi_parameter_sweep(base_design_configuration: dict, parameter_sweep_dictionary: dict)

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
           "parameter_3": 0
       }


.. py:function:: configure_cocotb_simulation(design_directory: str, simulator: Literal[icarus, verilator], top_level_language: Literal[verilog, vhdl], top_level_verilog_module: str, test_python_module: str, verilog_sources: list)

   Writes a cocotb makefile

   In the form:
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


.. py:data:: make_cocotb

   

.. py:function:: run_cocotb_simulation(design_directory: str)

   Equivalent to running the cocotb makefile.


.. py:data:: write_cocotb_makefile

   

