:py:mod:`piel.visual`
=====================

.. py:module:: piel.visual


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   auto_plot_multiple/index.rst
   data_conversion/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.visual.plot_simple_multi_row
   piel.visual.plot_multi_row
   piel.visual.append_row_to_dict
   piel.visual.points_to_lines_fixed_transient



.. py:function:: plot_simple_multi_row(data: pandas.DataFrame, x_axis_column_name: str = 't', row_list: list | None = None, y_axis_title_list: list | None = None, x_axis_title: str | None = None)

   Plot multiple rows of data on the same plot. Each row is a different line. Each row is a different y axis. The x
   axis is the same for all rows. The y axis title is the same for all rows.

   :param data: Data to plot.
   :type data: pd.DataFrame
   :param x_axis_column_name: Column name of the x axis. Defaults to "t".
   :type x_axis_column_name: str, optional
   :param row_list: List of column names to plot. Defaults to None.
   :type row_list: list, optional
   :param y_axis_title_list: List of y axis titles. Defaults to None.
   :type y_axis_title_list: list, optional
   :param x_axis_title: Title of the x axis. Defaults to None.
   :type x_axis_title: str, optional

   :returns: Matplotlib plot.
   :rtype: plt


.. py:function:: plot_multi_row(data: pandas.DataFrame)


.. py:function:: append_row_to_dict(data: dict, copy_index: int, set_value: dict)

   Get all the rows of the dictionary. We want to copy and append a row at a particular index of the dictionary values.
   Operates on existing data

   :param data: Dictionary of data to be appended.
   :param copy_index: Index of the row to be copied.
   :param set_value: Dictionary of values to be set at the copied index.

   :returns: None


.. py:function:: points_to_lines_fixed_transient(data: pandas.DataFrame | dict, time_index_name: str, fixed_transient_time=1, return_dict: bool = False)

   This function converts specific steady-state point data into steady-state lines with a defined transient time in order to plot digital-style data.

   For example, VCD data tends to be structured in this form:

   .. code-block:: text

       #2001
       b1001 "
       b10010 #
       b1001 !
       #4001
       b1011 "
       b1011 #
       b0 !
       #6001
       b101 "

   This means that even when tokenizing the data, when visualising it in a wave plotter such as GTKWave, the signals
   get converted from token specific times to transient signals by a corresponding transient rise time. If we want
   to plot the data correspondingly in Python, it is necessary to add some form of transient signal translation.
   Note that this operates on a dataframe where the electrical time signals are clearly defined. It copies the
   corresponding steady-state data points whilst adding data points for the time-index accordingly.

   It starts by creating a copy of the initial dataframe as to not overwrite the existing data. We have an initial
   time data point that tends to start at time 0. This means we need to add a point just before the next steady
   state point transition. So what we want to do is copy the existing row and just change the time to be the
   `fixed_transient_time` before the next transition.

   Doesn't append on penultimate row.

   :param data: Dataframe or dictionary of data to be converted.
   :param time_index_name: Name of the time index column.
   :param fixed_transient_time: Time of the transient signal.
   :param return_dict: Return a dictionary instead of a dataframe.

   :returns: Dataframe or dictionary of data with steady-state lines.
