import matplotlib.pyplot as plt
from ..types import PropagationDelayMeasurementSweepData
from typing import Optional


def plot_signal_propagation_sweep_signals(
    signal_propagation_sweep_data: PropagationDelayMeasurementSweepData,
    measurement_section: Optional[list[str]] = None,
    xlabel=r"Time $ns$",
    ylabel=r"Voltage $mV$",
):
    fig, axs = plt.subplots(
        len(signal_propagation_sweep_data.data),
        1,
        sharex=True,
    )
    axs[0].set_xlim(
        [
            signal_propagation_sweep_data.data[1].reference_waveform.time_s[0],
            signal_propagation_sweep_data.data[1].reference_waveform.time_s[-1],
        ]
    )

    reference_x_data = list()
    reference_y_data = list()
    dut_x_data = list()
    dut_y_data = list()

    i = 0
    for signal_propagation_measurement_data_i in signal_propagation_sweep_data.data:
        # Go through each of the files measurements to extract the relevant files
        reference_x_data = (
            signal_propagation_measurement_data_i.reference_waveform.time_s
        )
        reference_y_data = signal_propagation_measurement_data_i.reference_waveform.data
        dut_x_data = signal_propagation_measurement_data_i.dut_waveform.time_s
        dut_y_data = signal_propagation_measurement_data_i.dut_waveform.data
        ax = axs[i]

        ax.plot(
            reference_x_data,
            reference_y_data,
            "-",
            label=f"reference_{signal_propagation_measurement_data_i.reference_waveform.data_name}",
        )
        ax.plot(
            dut_x_data,
            dut_y_data,
            "-",
            label=f"reference_{signal_propagation_measurement_data_i.dut_waveform.data_name}",
        )
        i += 1

    # ax.legend()
    # fig.set_title("Transient Propagation Delay Characterization \n RF PCB")
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)

    return fig, ax


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
