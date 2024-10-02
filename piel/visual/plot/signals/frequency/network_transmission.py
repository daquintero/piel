from typing import Optional, Tuple
import numpy as np
from piel.types import NetworkTransmission
from piel.visual.plot.basic import plot_simple
from piel.visual.plot.position import create_axes_per_figure
import matplotlib.figure as mpl_fig
import matplotlib.axes as mpl_axes
import logging

logger = logging.getLogger(__name__)


def plot_two_port_gain_in_dBm(
    frequency_array_state: NetworkTransmission,
    fig: Optional[mpl_fig.Figure] = None,
    axs: Optional[mpl_axes.Axes] = None,
    label: Optional[str] = None,
) -> Tuple[mpl_fig.Figure, mpl_axes.Axes]:
    """
    Plots input power (p_in_dbm) vs S21 gain (s_21_db) from a NetworkTransmission object.

    Parameters:
    -----------
    frequency_array_state : NetworkTransmission
        The NetworkTransmission object containing the measurement data.

    fig : matplotlib.figure.Figure, optional
        The figure object to plot on. If None, a new figure is created.

    axs : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, a new set of axes is created.

    label : str, optional
        The label for the plot. If None, a default label is used.

    Returns:
    --------
    tuple
        A tuple containing the matplotlib Figure and Axes objects.
    """
    # Create axes if not provided
    if (fig is None) or (axs is None):
        fig, axs = create_axes_per_figure(rows=1, columns=1)

    # Extract input power in dBm from ScalarSource.phasor.magnitude
    try:
        p_in_dbm = np.array(frequency_array_state.input.phasor.magnitude)
    except AttributeError as e:
        logger.error(
            "Failed to extract 'p_in_dbm' from NetworkTransmission.input.phasor.magnitude."
        )
        raise e

    # Initialize s_21_db as None
    s_21_db = None

    # Iterate through network transmissions to find S21
    for path_transmission in frequency_array_state.network:
        if path_transmission.ports == ("in0", "out0"):
            # Compute magnitude in dB from complex transmission
            transmission = np.array(path_transmission.transmission)
            # Avoid log of zero by adding a small epsilon
            epsilon = 1e-12
            s_21_db = 20 * np.log10(np.abs(transmission) + epsilon)
            break

    if s_21_db is None:
        logger.error(
            "S21 transmission ('in0', 'out0') not found in NetworkTransmission.network."
        )
        raise ValueError("S21 transmission ('in0', 'out0') not found.")

    # Determine label
    plot_label = label if label is not None else "S21 Gain"

    # Plot the data
    fig, axs = plot_simple(
        p_in_dbm,
        s_21_db,
        fig=fig,
        axs=axs,
        label=plot_label,
        xlabel=r"$P_{in}$ $dBm$",
        ylabel=r"$S_{21}$ $dB$",
    )

    return fig, axs
