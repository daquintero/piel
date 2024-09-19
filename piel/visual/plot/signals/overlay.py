from piel.types import MultiDataTimeSignal
import numpy as np
import matplotlib.pyplot as plt


def plot_multi_data_time_signal(
    multi_signal: MultiDataTimeSignal, figsize: tuple = (10, 6)
):
    """
    Plots all rising edge signals on the same figure with a shared x-axis.

    Args:
        multi_signal (List[DataTimeSignalData]): List of rising edge signals.
        figsize (tuple): Size of the plot (width, height) in inches. Default is (10, 6).

    Returns:
        None
    """
    if not multi_signal:
        raise ValueError("The multi_signal list is empty.")

    plt.figure(figsize=figsize)

    for signal in multi_signal:
        if not signal.time_s:
            raise ValueError(f"Signal '{signal.data_name}' has an empty time_s array.")

        time = np.array(signal.time_s)
        data = np.array(signal.data)

        plt.plot(time, data, label=signal.data_name)

    # Remove axis labels
    plt.xlabel("")
    plt.ylabel("")

    # Optionally, remove ticks
    plt.xticks([])
    plt.yticks([])

    # Optionally, remove the legend if labels are not needed
    plt.legend().set_visible(False)

    # Add grid if desired
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.title("Rising Edge Signals")
    plt.show()
