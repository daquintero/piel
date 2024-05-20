:py:mod:`piel.tools.cocotb.core`
================================

.. py:module:: piel.tools.cocotb.core

.. autoapi-nested-parse::

   The objective of this file is to provide the simulation ports and interconnection to consider modelling digital and mixed signal logic.

   The main simulation driver is cocotb, and this generates a set of files that correspond to time-domain digital
   simulations. The cocotb verification software can also be used to perform mixed signal simulation, and digital data
   can be inputted as a bitstream into a photonic solver, although the ideal situation would be to have integrated
   photonic time-domain models alongside the electronic simulation solver, and maybe this is where it will go. It can be
   assumed that, as is currently, cocotb can interface python with multiple solvers until someone (and I'd love to do
   this) writes an equivalent python-based or C++ based python time-domain simulation solver.

   The nice thing about cocotb is that as long as the photonic simulations can be written asynchronously, time-domain
   simulations can be closely integrated or simulated through this verification software.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.cocotb.core.check_cocotb_testbench_exists
   piel.tools.cocotb.core.configure_cocotb_simulation
   piel.tools.cocotb.core.run_cocotb_simulation



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.tools.cocotb.core.delete_simulation_output_files


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
