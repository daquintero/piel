:py:mod:`piel.openlane.v1_parse.openlane_opensta_v1`
====================================================

.. py:module:: piel.openlane.v1_parse.openlane_opensta_v1

.. autoapi-nested-parse::

   These functions do not work currently.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.openlane.v1_parse.openlane_opensta_v1.read_file_meta_data
   piel.openlane.v1_parse.openlane_opensta_v1.configure_frame_id
   piel.openlane.v1_parse.openlane_opensta_v1.configure_timing_data_rows
   piel.openlane.v1_parse.openlane_opensta_v1.extract_frame_meta_data
   piel.openlane.v1_parse.openlane_opensta_v1.extract_timing_data
   piel.openlane.v1_parse.openlane_opensta_v1.calculate_propagation_delay
   piel.openlane.v1_parse.openlane_opensta_v1.run_parser



.. py:function:: read_file_meta_data(file_path: str | pathlib.Path) -> pandas.DataFrame

   Read the file and extract the metadata

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

   :returns: Dataframe containing the file lines
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: configure_frame_id(file_lines_data)


.. py:function:: configure_timing_data_rows(file_lines_data, maximum_frame_amount)


.. py:function:: extract_frame_meta_data(file_lines_data)


.. py:function:: extract_timing_data(file_address, frame_meta_data, frame_id=0)


.. py:function:: calculate_propagation_delay(net_name_in, net_name_out, timing_data)


.. py:function:: run_parser(file_address)
