:py:mod:`piel.tools.openlane.v2`
================================

.. py:module:: piel.tools.openlane.v2


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.openlane.v2.get_all_designs_metrics_openlane_v2
   piel.tools.openlane.v2.read_metrics_openlane_v2
   piel.tools.openlane.v2.run_openlane_flow



.. py:function:: get_all_designs_metrics_openlane_v2(output_directory: piel.types.PathTypes, target_prefix: str)

   Returns a dictionary of all the metrics for all the designs in the output directory.

   Usage:

       ```python
       from piel.tools.openlane import get_all_designs_metrics_v2

       metrics = get_all_designs_metrics_v2(
           output_directory="output",
           target_prefix="design",
       )
       ```

   :param output_directory: The path to the output directory.
   :type output_directory: PathTypes
   :param target_prefix: The prefix of the designs to get the metrics for.
   :type target_prefix: str

   :returns: A dictionary of all the metrics for all the designs in the output directory.
   :rtype: dict


.. py:function:: read_metrics_openlane_v2(design_directory: piel.types.PathTypes) -> dict

   Read design metrics from OpenLane v2 run files.

   :param design_directory: Design directory PATH.
   :type design_directory: PathTypes

   :returns: Metrics dictionary.
   :rtype: dict


.. py:function:: run_openlane_flow(configuration: dict | None = None, design_directory: piel.types.PathTypes = '.', parallel_asynchronous_run: bool = False, only_generate_flow_setup: bool = False)

   Runs the OpenLane v2 flow.

   :param configuration: OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: PathTypes
   :param parallel_asynchronous_run: Run the flow in parallel.
   :type parallel_asynchronous_run: bool
   :param only_generate_flow_setup: Only generate the flow setup.
   :type only_generate_flow_setup: bool

   Returns:
