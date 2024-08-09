"""
We want to streamline the figure and axes generation based on a given list which contains the data to be plotted.
Each component, as well,
may require more than one plot or a given set of plots. So it makes sense to both generalize this as a creation of plots
per a given set of parameters,
 which may or not be linked to the number of data points in a given list. The other complexity is the given structure
 of the axes for a given figure.

There can be multiple elements per plot. There are overlaying plots and separate plots.
Overlaying plots require sharing the same axes and separate plots require the same figure.
So, we want to configure plotting types based on this.
The question in this case, is of combining multiple figures, or just combining multiple axes.
It sounds like creating the axes is the best way to implement this for a given figure.
"""

import matplotlib.pyplot as plt
from ..types import AxesPlottingTypes, ExtensiblePlotsDirectionPerElement


def create_axes_per_figure(rows: int = 1, columns: int = 1, **kwargs) -> tuple:
    """
    This function creates a figure and a set of axes in this figure according to the number of rows or columns defined.
    """
    fig, axs = plt.subplots(rows, columns, **kwargs)

    if (rows == 1) and (columns == 1):
        # We always want this to be an array so we can compose easily with the rest of the code.
        axs = [axs]

    return fig, axs


def list_to_separate_plots(
    container_list: list,
    axes_per_element: int = 1,
    multi_axes_extension_direction: ExtensiblePlotsDirectionPerElement = "x",
    **kwargs,
) -> tuple:
    """
    This function creates a list of plots that are separate from each other.
    """
    elements_amount = len(container_list)

    if (axes_per_element > 1) and (multi_axes_extension_direction == "x"):
        rows = elements_amount
        columns = axes_per_element
    elif (axes_per_element > 1) and (multi_axes_extension_direction == "y"):
        rows = elements_amount * axes_per_element
        columns = 1
    else:
        rows = elements_amount
        columns = axes_per_element

    fig, axs = create_axes_per_figure(rows=rows, columns=columns, **kwargs)
    return fig, axs


def list_to_overlayed_plots(container_list: list, **kwargs) -> tuple:
    fig, axs = create_axes_per_figure(rows=1, columns=1, **kwargs)
    return fig, axs


def create_plot_containers(
    container_list: list, axes_structure: AxesPlottingTypes = "separate", **kwargs
) -> tuple:
    if axes_structure == "separate":
        fig, axs = list_to_separate_plots(container_list=container_list, **kwargs)
    elif axes_structure == "overlay":
        fig, axs = list_to_overlayed_plots(container_list=container_list, **kwargs)
    return fig, axs
