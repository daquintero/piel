from typing import Any
import numpy as np
from piel.types import FrequencyTransmissionArrayState
from piel.visual.plot.basic import plot_simple, create_axes_per_figure


def plot_frequency_array_state_power_in_s21_db(
    frequency_array_state: FrequencyTransmissionArrayState,
    fig: Any = None,
    axs: Any = None,
) -> tuple:
    if (fig is None) or (axs is None):
        fig, axs = create_axes_per_figure(rows=1, columns=1)

    power_in = np.array(frequency_array_state.p_in_dbm)
    gain = np.array(frequency_array_state.s_21_db)

    fig, axs = plot_simple(
        power_in,
        gain,
        fig=fig,
        axs=axs,
        label=frequency_array_state.name,
        xlabel=r"$P_{in}$ $dBm$",
        ylabel=r"$S_{21}$ $dB$",
    )

    return fig, axs
