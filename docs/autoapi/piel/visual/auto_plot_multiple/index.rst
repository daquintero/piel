:py:mod:`piel.visual.auto_plot_multiple`
========================================

.. py:module:: piel.visual.auto_plot_multiple


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.visual.auto_plot_multiple.plot_simple
   piel.visual.auto_plot_multiple.plot_simple_multi_row



.. py:function:: plot_simple(x_data: numpy.array, y_data: numpy.array, label: str | None = None, ylabel: str | None = None, xlabel: str | None = None, fig: matplotlib.pyplot.Figure | None = None, ax: matplotlib.pyplot.Axes | None = None, *args, **kwargs)

   Plot a simple line graph. The desire of this function is just to abstract the most basic data representation whilst
   keeping the flexibility of the matplotlib library. The goal would be as well that more complex data plots can be
   constructed from a set of these methods.

   :param x_data: X axis data.
   :type x_data: np.array
   :param y_data: Y axis data.
   :type y_data: np.array
   :param label: Label for the plot. Defaults to None.
   :type label: str, optional
   :param ylabel: Y axis label. Defaults to None.
   :type ylabel: str, optional
   :param xlabel: X axis label. Defaults to None.
   :type xlabel: str, optional
   :param fig: Matplotlib figure. Defaults to None.
   :type fig: plt.Figure, optional
   :param ax: Matplotlib axes. Defaults to None.
   :type ax: plt.Axes, optional

   :returns: Matplotlib plot.
   :rtype: plt


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


