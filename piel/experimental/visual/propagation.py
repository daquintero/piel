import matplotlib.pyplot as plt
from ...visual import save
from ..types import ExperimentData
from typing import Optional


def plot_signal_propagation_signals(
    experiment_data: ExperimentData,
    measurement_section: Optional[list[str]] = None,
    xlabel=r"Time $ns$",
    ylabel=r"Voltage $mV$",
    *args,
    **kwargs,
):
    # TODO Implement validation that it's a time-propagation delay measurement
    signal_propagation_sweep_data = experiment_data.data.collection

    fig, axs = plt.subplots(
        len(signal_propagation_sweep_data),
        1,
        sharex=True,
    )
    axs[0].set_xlim(
        [
            signal_propagation_sweep_data[1].reference_waveform.time_s[0],
            signal_propagation_sweep_data[1].reference_waveform.time_s[-1],
        ]
    )

    reference_x_data = list()
    reference_y_data = list()
    dut_x_data = list()
    dut_y_data = list()

    i = 0
    for signal_propagation_measurement_data_i in signal_propagation_sweep_data:
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

    if kwargs["path"]:
        save(fig, **kwargs)

    return fig, ax


def plot_signal_propagation_measurements(
    experiment_data: ExperimentData,
    x_parameter: str,
    measurement_name: str,
    measurement_section: Optional[list[str]] = None,
    xlabel=r"Source Frequency $GHz$",
    ylabel=r"Propagation Delay $ns$",
    yscale_factor=1e9,
    *args,
    **kwargs,
):
    import pandas as pd

    x_parameter_data = pd.DataFrame(experiment_data.experiment.parameters_list)[
        x_parameter
    ]

    signal_propagation_sweep_data = experiment_data.data.collection

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if measurement_section is None:
        measurement_section = ["value", "mean", "min", "max"]

    for measurement_section_i in measurement_section:
        x_data = list()
        y_data = list()
        i = 0
        for signal_propagation_measurement_data_i in signal_propagation_sweep_data:
            # Go through each of the files measurements to extract the relevant files
            x_data.append(x_parameter_data[i])
            y_data.append(
                getattr(
                    signal_propagation_measurement_data_i.measurements[
                        measurement_name
                    ],
                    measurement_section_i,
                )
                * yscale_factor
            )
            i += 1

        ax.plot(x_data, y_data, "o", label=measurement_section_i)

    ax.legend()
    ax.set_title("Transient Propagation Delay Characterization \n RF PCB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, ax
