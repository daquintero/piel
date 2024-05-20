:py:mod:`piel.tools.amaranth.verify`
====================================

.. py:module:: piel.tools.amaranth.verify


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.amaranth.verify.verify_truth_table



.. py:function:: verify_truth_table(truth_table_amaranth_module: amaranth.Elaboratable, truth_table_dictionary: dict, inputs: list, outputs: list, vcd_file_name: str, target_directory: piel.types.PathTypes, implementation_type: Literal[combinatorial, sequential, memory] = 'combinatorial')

   We will implement a function that tests the module to verify that the outputs generates match the truth table provided.

   TODO Implement a similar function from the openlane netlist too.
   TODO unclear they can implement verification without it being in a synchronous simulation.
