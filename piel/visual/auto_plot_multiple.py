import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

matplotlib.style.use(pathlib.Path(__file__) / ".." / pathlib.Path("piel_fast.rcParams"))

__all__ = [
    "plot_simple",
    "plot_simple_multi_row",
]


def plot_simple(x_data: np.array, y_data: np.array, ylabel: str, xlabel: str):
    """
    Plot a simple line graph.

    Args:
        x_data (np.array): X axis data.
        y_data (np.array): Y axis data.
        ylabel (str): Y axis label.
        xlabel (str): X axis label.

    Returns:
        plt: Matplotlib plot.
    """
    plt.plot(x_data, y_data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return plt


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
