:py:mod:`piel.visual.auto_plot_multiple`
========================================

.. py:module:: piel.visual.auto_plot_multiple


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.visual.auto_plot_multiple.plot_simple_multi_row
   piel.visual.auto_plot_multiple.plot_multi_row



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
