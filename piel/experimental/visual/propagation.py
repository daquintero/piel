import matplotlib.pyplot as plt
from ..types import PropagationDelayMeasurementSweepData
from typing import Optional


def plot_signal_propagation_sweep_measurement(
    signal_propagation_sweep_data: PropagationDelayMeasurementSweepData,
    measurement_name: str,
    measurement_section: Optional[list[str]] = None,
    xlabel=r"Source Frequency $GHz$",
    ylabel=r"Propagation Delay $ns$",
    yscale_factor=1e9,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if measurement_section is None:
        measurement_section = ["value", "mean", "min", "max"]

    for measurement_section_i in measurement_section:
        x_data = list()
        y_data = list()
        for signal_propagation_measurement_data_i in signal_propagation_sweep_data.data:
            # Go through each of the files measurements to extract the relevant files
            x_data.append(signal_propagation_measurement_data_i.name)
            y_data.append(
                getattr(
                    signal_propagation_measurement_data_i.measurements[
                        measurement_name
                    ],
                    measurement_section_i,
                )
                * yscale_factor
            )

        ax.plot(x_data, y_data, "o", label=measurement_section_i)

    ax.legend()
    ax.set_title("Transient Propagation Delay Characterization \n RF PCB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax
