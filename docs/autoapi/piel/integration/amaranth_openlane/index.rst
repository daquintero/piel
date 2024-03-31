:py:mod:`piel.integration.amaranth_openlane`
============================================

.. py:module:: piel.integration.amaranth_openlane

.. autoapi-nested-parse::

   This file enhances some functions that easily translates between an `amaranth` function to implement a `openlane` flow.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.amaranth_openlane.layout_amaranth_truth_table_through_openlane



.. py:function:: layout_amaranth_truth_table_through_openlane(amaranth_module: amaranth.Module, inputs_name_list: list[str], outputs_name_list: list[str], parent_directory: piel.types.PathTypes, target_directory_name: Optional[str] = None, openlane_version: Literal[v1, v2] = 'v2', **kwargs)

   This function implements an amaranth truth-table module through the openlane flow. There are several ways to
   implement a module. Fundamentally, this requires the verilog files to be generated from the openlane-module in a
   particular directory. For the particular directory provided, this function will generate the verilog files in the
   corresponding directory. It can also generate the ``openlane`` configuration files for this particular location.

   This function does a few things:

   1. Starts off from a ``amaranth`` module class.
   2. Determines the output directory in which to generate the files, and creates one accordingly if it does not exist.
   3. Generates the verilog files from the ``amaranth`` module class.
   4. Generates the ``openlane`` configuration files for this particular location.
   5. Implements the ``openlane`` flow for this particular location to generate a chip.

   :param amaranth_module: Amaranth module class.
   :type amaranth_module: amaranth.Module
   :param inputs_name_list: List of input names.
   :type inputs_name_list: list[str]
   :param outputs_name_list: List of output names.
   :type outputs_name_list: list[str]
   :param parent_directory: Parent directory PATH.
   :type parent_directory: PathTypes
   :param target_directory_name: Target directory name. If none is provided, it will default to the name of the amaranth elaboratable class.
   :type target_directory_name: Optional[str]
   :param openlane_version: OpenLane version. Defaults to ``v1``.
   :type openlane_version: Literal["v1", "v2"]

   :returns: None


