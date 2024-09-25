import numpy as np
import pandas as pd
from typing import List, Optional, Any
from .position import create_axes_per_figure

__all__ = [
    "plot_simple",
    "plot_simple_multi_row",
]


def plot_simple(
    x_data: np.ndarray,
    y_data: np.ndarray,
    label: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    fig: Optional[Any] = None,
    axs: Optional[list[Any]] = None,
    title: Optional[str] = None,
    plot_args: list = None,
    plot_kwargs: dict = None,
    figure_kwargs: dict = None,
    legend_kwargs: dict = None,
    *args,
    **kwargs,
) -> tuple:
    """
    Plot a simple line graph. This function abstracts the basic files representation while
    keeping the flexibility of the matplotlib library.

    Args:
        x_data (np.ndarray): X axis files.
        y_data (np.ndarray): Y axis files.
        label (Optional[str], optional): Label for the plot. Defaults to None.
        ylabel (Optional[str], optional): Y axis label. Defaults to None.
        xlabel (Optional[str], optional): X axis label. Defaults to None.
        fig (Optional[plt.Figure], optional): Matplotlib figure. Defaults to None.
        axs (Optional[list[plt.Axes]], optional): Matplotlib axes. Defaults to None.
        title (Optional[str], optional): Title of the plot. Defaults to None.
        *args: Additional arguments passed to plt.plot().
        **kwargs: Additional keyword arguments passed to plt.plot().

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axes of the plot.
    """

    if figure_kwargs is None:
        figure_kwargs = {
            "tight_layout": True,
        }

    if fig is None and axs is None:
        fig, axs = create_axes_per_figure(rows=1, columns=1, **figure_kwargs)

    if plot_kwargs is None:
        if label is not None:
            plot_kwargs = {"label": label}
        else:
            plot_kwargs = {}

    if plot_args is None:
        plot_args = list()

    ax = axs[0]
    ax.plot(x_data, y_data, *plot_args, **plot_kwargs)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)

    if (label is not None) and (legend_kwargs is not None):
        ax.legend(**legend_kwargs)

    # Rotate x-axis labels for better fit
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    return fig, axs


def plot_simple_multi_row(
    data: pd.DataFrame,
    x_axis_column_name: str = "t",
    row_list: Optional[List[str]] = None,
    y_label: Optional[List[str]] = None,
    x_label: Optional[str] = None,
    titles: Optional[List[str]] = None,
    subplot_spacing: float = 0.15,
) -> Any:
    """
    Plot multiple rows of files on separate subplots, sharing the same x-axis.

    Args:
        data (pd.DataFrame): Data to plot.
        x_axis_column_name (str, optional): Column name of the x-axis. Defaults to "t".
        row_list (Optional[List[str]], optional): List of column names to plot. Defaults to None.
        y_label (Optional[List[str]], optional): List of Y-axis titles for each subplot. Defaults to None.
        x_label (Optional[str], optional): Title of the x-axis. Defaults to None.
        titles (Optional[List[str]], optional): Titles for each subplot. Defaults to None.
        subplot_spacing (float, optional): Spacing between subplots. Defaults to 0.3.

    Returns:
        plt.Figure: The matplotlib figure containing the subplots.
    """
    import matplotlib.pyplot as plt

    if row_list is None:
        raise ValueError("row_list must be provided")

    x_data = data[x_axis_column_name]
    y_data_list = [data[row] for row in row_list]

    if y_label is None:
        y_label = row_list

    if titles is None:
        titles = [""] * len(row_list)

    row_amount = len(row_list)
    fig, axes = plt.subplots(row_amount, 1, sharex=True, figsize=(8, row_amount * 2))

    if row_amount == 1:
        axes = [axes]

    for _, (ax_i, y_data_i, y_label_i, title) in enumerate(
        zip(axes, y_data_list, y_label, titles)
    ):
        ax_i.plot(x_data, y_data_i)
        ax_i.grid(True)
        ax_i.set_ylabel(y_label_i)
        ax_i.set_title(title)

    if x_label is not None:
        axes[-1].set_xlabel(x_label)

    # Rotate x-axis labels for better fit
    for label in axes[-1].get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    fig.tight_layout()
    plt.subplots_adjust(hspace=subplot_spacing)  # Add space between subplots

    return fig
