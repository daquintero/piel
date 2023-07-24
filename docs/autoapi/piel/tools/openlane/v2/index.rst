:py:mod:`piel.tools.openlane.v2`
================================

.. py:module:: piel.tools.openlane.v2


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.openlane.v2.run_openlane_flow



.. py:function:: run_openlane_flow(configuration: dict | None = test_spm_open_lane_configuration, design_directory: piel.config.piel_path_types = '/foss/designs/spm') -> None

   Runs the OpenLane flow.

   :param configuration: OpenLane configuration dictionary. If none is present it will default to the config.json file on the design_directory.
   :type configuration: dict
   :param design_directory: Design directory PATH.
   :type design_directory: piel_path_types

   :returns: None
