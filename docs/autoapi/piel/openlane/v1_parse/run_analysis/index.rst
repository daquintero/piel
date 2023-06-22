:py:mod:`piel.openlane.v1_parse.run_analysis`
=============================================

.. py:module:: piel.openlane.v1_parse.run_analysis

.. autoapi-nested-parse::

   TODO these functions do not currently work.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.openlane.v1_parse.run_analysis.get_all_rpt_files
   piel.openlane.v1_parse.run_analysis.extract_metrics_timing
   piel.openlane.v1_parse.run_analysis.run_analysis



.. py:function:: get_all_rpt_files(run_directory=None)


.. py:function:: extract_metrics_timing(timing_sta_files_list)

   For every file in the sta timing file, extract the propagation delay and save the file meta data into a dictionary.


.. py:function:: run_analysis(run_directory)
