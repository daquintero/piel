from piel.visual import save, create_plot_containers
from piel.experimental.types import PropagationDelayMeasurementDataCollection
from typing import Optional


def plot_propagation_signals_time(
    data_collection: PropagationDelayMeasurementDataCollection,
    measurement_section: Optional[list[str]] = None,
    xlabel=r"Time $ns$",
    ylabel=r"Voltage $mV$",
    *args,
    **kwargs,
):
    # TODO Implement validation that it's a time-propagation delay measurement
    signal_propagation_sweep_data = data_collection.collection

    fig, axs = create_plot_containers(
        signal_propagation_sweep_data,
        sharex=True,
    )

    # Manage missing data here
    if signal_propagation_sweep_data[1].reference_waveform is None:
        pass
    else:
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
        if signal_propagation_measurement_data_i.reference_waveform is None:
            pass
        else:
            # Go through each of the files measurements to extract the relevant files
            reference_x_data = (
                signal_propagation_measurement_data_i.reference_waveform.time_s
            )
            reference_y_data = (
                signal_propagation_measurement_data_i.reference_waveform.data
            )
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

    save(fig, **kwargs)

    return fig, axs
