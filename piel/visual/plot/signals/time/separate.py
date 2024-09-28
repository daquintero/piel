from typing import Any
from piel.types import MultiDataTimeSignal, Unit
import numpy as np
from piel.visual.plot.position import create_axes_per_figure
from piel.visual.plot.core import save
import logging

logger = logging.getLogger(__name__)


def plot_multi_data_time_signal_different(
    multi_signal: MultiDataTimeSignal,
    fig: Any = None,
    axs: Any = None,
    subplots_kwargs: dict = None,
    xlabel: str | Unit | list[Unit] = None,
    ylabel: str | Unit | list[Unit] | list = None,
    title: str | Unit | list = None,
    **kwargs,
):
    """
    Plots all rising edge signals on the same figure, but with a shared x-axis, but multiple y axes.

    Args:
        multi_signal (List[DataTimeSignalData]): List of rising edge signals.
        subplots_kwargs (dict): Keyword arguments to pass to create_axes_per_figure.

    Returns:
        None
    """
    signal_amount = len(multi_signal)

    x_correction = 1
    y_correction = np.repeat(1, signal_amount)

    if not multi_signal:
        raise ValueError("The multi_signal list is empty.")

    if xlabel is None:
        xlabel = r"Time $s$"
    elif isinstance(xlabel, str):
        pass
    elif isinstance(xlabel, Unit):
        x_correction = xlabel.base
        logger.warning(
            f"Data correction of 1/{x_correction} from unit definition {xlabel} will be applied on x-axis"
        )
        xlabel = xlabel.label

    if ylabel is None:
        ylabel = np.repeat(r"Voltage $V$", signal_amount)
    elif isinstance(ylabel, str):
        pass
    elif isinstance(ylabel, list):
        pass
    elif isinstance(ylabel, Unit):
        y_correction = ylabel.base
        logger.warning(
            f"Data correction of 1/{y_correction} from unit definition {ylabel} will be applied on all y-axis."
        )
        ylabel = ylabel.label
    elif isinstance(ylabel, list):
        # THis should be a list of units
        i = 0
        for unit_i in ylabel:
            if isinstance(unit_i, Unit):
                y_correction[i] = unit_i.base

            i += 1

    if subplots_kwargs is None:
        subplots_kwargs = {}

    if (fig is None) or (axs is None):
        fig, axs = create_axes_per_figure(
            rows=len(multi_signal), columns=1, **subplots_kwargs
        )

    if title is None:
        pass
    elif title is str:
        fig.suptitle(title)

    i = 0
    for signal in multi_signal:
        if (len(signal.time_s) == 0) or (signal.time_s is None):
            raise ValueError(f"Signal '{signal.data_name}' has an empty time_s array.")

        time = np.array(signal.time_s) / x_correction
        data = np.array(signal.data) / y_correction[i]

        axs[i].plot(
            time,
            data,
            label=signal.data_name,
            # color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i],
        )
        # print(i)

        axs[i].set_ylabel(ylabel[i])

        if isinstance(title, list):
            axs[i].set_title(title[i], loc="left")

        i += 1

    fig.supxlabel(xlabel)

    save(fig, **kwargs)

    return fig, axs
