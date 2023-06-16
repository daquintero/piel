:py:mod:`piel.simulation`
=========================

.. py:module:: piel.simulation

.. autoapi-nested-parse::

   The objective of this file is to provide the simulation ports and interconnection to consider modelling digital and mixed signal logic.

   The main simulation driver is cocotb, and this generates a set of files that correspond to time-domain digital simulations.
   The cocotb verification software can also be used to perform mixed signal simulation, and digital data can be inputted as a bitstream into a photonic solver, although the ideal situation would be to have integrated photonic time-domain models alongside the electronic simulation solver, and maybe this is where it will go. It can be assumed that, as is currently, cocotb can interface python with multiple solvers until someone (and I'd love to do this) writes an equivalent python-based or C++ based python time-domain simulation solver.

   The nice thing about cocotb is that as long as the photonic simulations can be writen asyncrhonously, time-domain simulations can be closesly integrated or simulated through this verification software.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.simulation.configure_cocotb_simulation
   piel.simulation.run_cocotb_simulation



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.simulation.write_cocotb_makefile
   piel.simulation.make_cocotb


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


.. py:function:: run_cocotb_simulation(design_directory: str)

   Equivalent to running the cocotb makefile.


.. py:data:: write_cocotb_makefile

   

.. py:data:: make_cocotb

   

