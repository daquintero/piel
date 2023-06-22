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



.. py:function:: read_file_meta_data(file_path: str | pathlib.Path)

   Read the file and extract the metadata

   :param file_path: Path to the file
   :type file_path: str | pathlib.Path

   :returns: Dataframe containing the file lines
             maximum_frame_amount (int): Maximum number of frames in the file
             frame_meta_data (dict): Dictionary containing the frame metadata
   :rtype: file_lines_data (pd.DataFrame)


.. py:function:: configure_frame_id(file_lines_data: pandas.DataFrame)

   Configure the frame id for each line in the file

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame

   :returns: Maximum number of frames in the file
             file_lines_data (pd.DataFrame): Dataframe containing the file lines
   :rtype: maximum_frame_amount (int)


.. py:function:: configure_timing_data_rows(file_lines_data, maximum_frame_amount)

   Configure the timing data rows for each frame in the file

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame
   :param maximum_frame_amount: Maximum number of frames in the file
   :type maximum_frame_amount: int

   :returns: Dictionary containing the frame metadata
   :rtype: frame_meta_data (dict)


.. py:function:: extract_frame_meta_data(file_lines_data)

   Extract the frame metadata

   :param file_lines_data: Dataframe containing the file lines
   :type file_lines_data: pd.DataFrame

   :returns: Dataframe containing the start point names
             end_point_name (pd.DataFrame): Dataframe containing the end point names
             path_group_name (pd.DataFrame): Dataframe containing the path group names
             path_type_name (pd.DataFrame): Dataframe containing the path type names
   :rtype: start_point_name (pd.DataFrame)


.. py:function:: extract_timing_data(file_address, frame_meta_data, frame_id=0)

   Extract the timing data

   :param file_address: Path to the file
   :type file_address: str | pathlib.Path

   :returns: Dataframe containing the timing data
   :rtype: timing_data (pd.DataFrame)


.. py:function:: calculate_propagation_delay(net_name_in, net_name_out, timing_data)

   Calculate the propagation delay

   :param net_name_in: Name of the input net
   :type net_name_in: str
   :param net_name_out: Name of the output net
   :type net_name_out: str
   :param timing_data: Dataframe containing the timing data
   :type timing_data: pd.DataFrame

   :returns: Dataframe containing the propagation delay
   :rtype: propagation_delay_dataframe (pd.DataFrame)


.. py:function:: run_parser(file_address)

   Run the parser

   :param file_address: Path to the file
   :type file_address: str | pathlib.Path

   :returns: Dictionary containing the frame timing data
             propagation_delay (dict): Dictionary containing the propagation delay
   :rtype: frame_timing_data (dict)
