:py:mod:`piel.tools.amaranth.construct`
=======================================

.. py:module:: piel.tools.amaranth.construct


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.amaranth.construct.construct_amaranth_module_from_truth_table



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


