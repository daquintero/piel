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

   This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

   :param run_directory: The run directory to perform the analysis on. Defaults to None.
   :type run_directory: str, optional

   :returns: List of all the .rpt files in the run directory.
             timing_sta_files_list (list): List of all the .rpt files in the run directory.
             power_sta_files_list (list): List of all the .rpt files in the run directory.
   :rtype: all_rpt_files_list (list)


.. py:function:: extract_metrics_timing(timing_sta_files_list)

   For every file in the sta timing file, extract the propagation delay and save the file meta data into a dictionary.

   :param timing_sta_files_list: List of all the .rpt files in the run directory.
   :type timing_sta_files_list: list

   :returns: List of dictionaries containing the file meta data and the propagation delay.
   :rtype: timing_metrics_list (list)


.. py:function:: run_analysis(run_directory)

   This function aims to list and perform analysis on all the relevant files in a particular run between all the corners.

   :param run_directory: The run directory to perform the analysis on. Defaults to None.
   :type run_directory: str, optional

   :returns: List of dictionaries containing the file meta data and the propagation delay.
   :rtype: timing_metrics_list (list)
