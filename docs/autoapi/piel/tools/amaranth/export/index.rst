:py:mod:`piel.tools.amaranth.export`
====================================

.. py:module:: piel.tools.amaranth.export


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.amaranth.export.generate_verilog_from_amaranth



.. py:function:: generate_verilog_from_amaranth(amaranth_module: amaranth.Elaboratable, ports_list: list[str], target_file_name: str, target_directory: piel.config.piel_path_types, backend=verilog) -> None

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
   :type target_directory: piel_path_types
   :param backend: Backend to use. Defaults to ``verilog``.
   :type backend: amaranth.back.verilog

   :returns: None


