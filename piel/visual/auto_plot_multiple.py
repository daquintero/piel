import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib



__all__ = [
    "plot_simple",
    "plot_simple_multi_row",
]


def plot_simple(
    x_data: np.array,
    y_data: np.array,
    label: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    *args,
    **kwargs
):
    """
    Plot a simple line graph. The desire of this function is just to abstract the most basic data representation whilst
    keeping the flexibility of the matplotlib library. The goal would be as well that more complex data plots can be
    constructed from a set of these methods.

    Args:
        x_data (np.array): X axis data.
        y_data (np.array): Y axis data.
        label (str, optional): Label for the plot. Defaults to None.
        ylabel (str, optional): Y axis label. Defaults to None.
        xlabel (str, optional): X axis label. Defaults to None.
        fig (plt.Figure, optional): Matplotlib figure. Defaults to None.
        ax (plt.Axes, optional): Matplotlib axes. Defaults to None.

    Returns:
        plt: Matplotlib plot.
    """
    if (ax is None) and (fig is None):
        fig, ax = plt.subplots()

    ax.plot(x_data, y_data, label=label, *args, **kwargs)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if label is not None:
        # This function appends to the existing plt legend
        ax.legend()

    return fig, ax


def plot_simple_multi_row(
    data: pd.DataFrame,
    x_axis_column_name: str = "t",
    row_list: list | None = None,
    y_axis_title_list: list | None = None,
    x_axis_title: str | None = None,
):
    """
    Plot multiple rows of data on the same plot. Each row is a different line. Each row is a different y axis. The x
    axis is the same for all rows. The y axis title is the same for all rows.

    Args:
        data (pd.DataFrame): Data to plot.
        x_axis_column_name (str, optional): Column name of the x axis. Defaults to "t".
        row_list (list, optional): List of column names to plot. Defaults to None.
        y_axis_title_list (list, optional): List of y axis titles. Defaults to None.
        x_axis_title (str, optional): Title of the x axis. Defaults to None.

    Returns:
        plt: Matplotlib plot.
    """
    x = data[x_axis_column_name]
    y_array = []

    if y_axis_title_list is None:
        y_axis_title_list = row_list

    row_amount = len(row_list)
    for row_name in row_list:
        y_array.append(data[row_name])

    fig, axes = plt.subplots(row_amount, 1, sharex=True)

    for i in range(len(row_list)):
        axes[i].plot(x, y_array[i])
        axes[i].grid(True)
        axes[i].set(ylabel=y_axis_title_list[i])

    # TODO Xaxis title
    # TODO align all ytitles

    return plt
