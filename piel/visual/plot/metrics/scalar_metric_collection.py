import numpy as np
from piel.visual.plot.position import create_axes_per_figure
from piel.visual.plot.core import save
from piel.types import Unit
from typing import Any
import logging

logger = logging.getLogger(__name__)


def plot_basic_scalar_metric_collection_component(
    scalar_metric_collection,
    metric_component: str,
    fig: Any = None,
    axs: Any = None,
    subplots_kwargs: dict = None,
    xlabel: str | Unit = None,
    ylabel: str | Unit = None,
    title: str | Unit = None,
    label: str | Unit = "",
    **kwargs,
):
    """
    Plots the specified component (e.g., 'mean', 'min', 'max') of a ScalarMetricCollection.

    Args:
        scalar_metric_collection (ScalarMetricCollection): The collection of scalar metrics to plot.
        metric_component (str): The component of the ScalarMetrics to plot (e.g., 'mean', 'min', 'max', 'value', 'standard_deviation', 'count').
        fig (matplotlib.figure.Figure, optional): Existing figure to plot on. If None, a new figure is created.
        axs (list[matplotlib.axes.Axes], optional): Existing list of axes to plot on. If None, new axes are created. Plots on [0] by default.
        subplots_kwargs (dict, optional): Keyword arguments to pass to create_axes_per_figure.
        xlabel (str | Unit, optional): Label for the x-axis.
        ylabel (str | Unit, optional): Label for the y-axis.
        title (str | Unit, optional): Title for the plot.
        **kwargs: Additional keyword arguments to pass to the save function.

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """

    # Extract the desired component from each metric in the collection
    component_values = []
    for metric in scalar_metric_collection.metrics:
        # Get the specified component (mean, min, max, etc.)
        component_value = getattr(metric, metric_component, None)
        if component_value is not None:
            component_values.append(component_value)
        else:
            component_values.append(float("nan"))  # Handle missing values

    # Handle empty or missing component values
    if len(component_values) == 0:
        raise ValueError(f"No valid {metric_component} values found in the collection.")

    # If no labels provided, set default values
    if xlabel is None:
        xlabel = "Metric Index"

    if ylabel is None:
        ylabel = scalar_metric_collection.metrics[0].unit.label
        y_correction = scalar_metric_collection.metrics[0].unit.base
    elif isinstance(ylabel, str):
        pass
    elif isinstance(ylabel, Unit):
        y_correction = ylabel.base
        logger.warning(
            f"Data correction of 1/{y_correction} from unit definition {ylabel} will be applied on y-axis."
        )
        ylabel = ylabel.label

    if subplots_kwargs is None:
        subplots_kwargs = {}

    if (fig is None) or (axs is None):
        fig, axs = create_axes_per_figure(rows=1, columns=1, **subplots_kwargs)
    elif fig is None or axs is None:
        raise ValueError("Both fig and ax should be provided together or left as None.")

    if title is None:
        pass
    else:
        fig.suptitle(title)

    ax = axs[0]

    # Plot the component values
    ax.plot(
        np.arange(len(component_values)),
        component_values,
        marker="o",
        linestyle="-",
        label=label,
    )

    # Set labels, title, and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Optional saving functionality
    save(fig, **kwargs)

    return fig, ax
