:py:mod:`piel.tools.amaranth`
=============================

.. py:module:: piel.tools.amaranth


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   construct/index.rst
   export/index.rst
   verify/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.amaranth.construct_amaranth_module_from_truth_table
   piel.tools.amaranth.generate_verilog_from_amaranth
   piel.tools.amaranth.verify_truth_table



.. py:function:: construct_amaranth_module_from_truth_table(truth_table: dict, inputs: list[str], outputs: list[str], implementation_type: Literal[combinatorial, sequential, memory] = 'combinatorial')

   This function implements a truth table as a module in amaranth,
   Note that in some form in amaranth each statement is a form of construction.

   The truth table is in the form of:

       detector_phase_truth_table = {
           "detector_in": ["00", "01", "10", "11"],
           "phase_map_out": ["00", "10", "11", "11"],
       }

   :param truth_table: The truth table in the form of a dictionary.
   :type truth_table: dict
   :param inputs: The inputs to the truth table.
   :type inputs: list[str]
   :param outputs: The outputs to the truth table.
   :type outputs: list[str]
   :param implementation_type: The type of implementation. Defaults to "combinatorial".
   :type implementation_type: Literal["combinatorial", "sequential", "memory"], optional

   :returns: Generated amaranth module.


.. py:function:: generate_verilog_from_amaranth(amaranth_module: amaranth.Elaboratable, ports_list: list[str], target_file_name: str, target_directory: piel.types.PathTypes, backend=verilog) -> None

   This function exports an amaranth module to either a defined path, or a project structure in the form of an
   imported multi-design module.

   Iterate over ports list and construct a list of references for the strings provided in ``ports_list``

   :param amaranth_module: Amaranth elaboratable class.
   :type amaranth_module: amaranth.Elaboratable
   :param ports_list: List of input names.
   :type ports_list: list[str]
   :param target_file_name: Target file name.
   :type target_file_name: str
   :param target_directory: Target directory PATH.
   :type target_directory: PathTypes
   :param backend: Backend to use. Defaults to ``verilog``.
   :type backend: amaranth.back.verilog

   :returns: None


.. py:function:: verify_truth_table(truth_table_amaranth_module: amaranth.Elaboratable, truth_table_dictionary: dict, inputs: list, outputs: list, vcd_file_name: str, target_directory: piel.types.PathTypes, implementation_type: Literal[combinatorial, sequential, memory] = 'combinatorial')

   We will implement a function that tests the module to verify that the outputs generates match the truth table provided.

   TODO Implement a similar function from the openlane netlist too.
   TODO unclear they can implement verification without it being in a synchronous simulation.


