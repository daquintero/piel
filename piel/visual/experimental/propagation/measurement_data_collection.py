from piel.visual import (
    save,
    create_plot_containers,
    create_axes_parameters_tables_separate,
)
from piel.types.experimental import PropagationDelayMeasurementDataCollection
from typing import Optional


def plot_propagation_signals_time(
    data_collection: PropagationDelayMeasurementDataCollection,
    parameters_list: list = None,
    measurement_section: Optional[list[str]] = None,
    xlabel=r"Time $ns$",
    ylabel=r"Voltage $mV$",
    *args,
    **kwargs,
):
    """
    Note that this plot is a set of a separate plots.
    """
    if parameters_list is None:
        parameters_list = range(len(data_collection.collection))

    # TODO Implement validation that it's a time-propagation delay measurement
    fig, axs = create_plot_containers(
        data_collection.collection,
        sharex=True,
    )

    signal_propagation_sweep_data = data_collection.collection

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

    parameter_tables_list = list()
    reference_x_data = list()
    reference_y_data = list()
    dut_x_data = list()
    dut_y_data = list()

    i = 0
    for signal_propagation_measurement_data_i in data_collection.collection:
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

            # ax.set_title(parameters_list[i])
            parameter_tables_list.append(parameters_list[i])

            ax.plot(
                reference_x_data,
                reference_y_data,
                "-",
                label="REF",
            )
            ax.plot(
                dut_x_data,
                dut_y_data,
                "-",
                label="DUT",
            )

            ax.legend(loc="center right")

            # ax.set_xlabel(xlabel)
            # ax.set_ylabel(ylabel)

        i += 1

    # ax.legend()
    fig.suptitle(data_collection.name)

    if parameters_list is not None:
        if len(parameters_list) == len(data_collection.collection):
            # Create the labels accordingly
            try:
                create_axes_parameters_tables_separate(
                    fig=fig, axs=axs, parameter_tables_list=parameter_tables_list
                )
            except Exception as e:
                if "debug" in kwargs and kwargs.get("debug", False):
                    raise e

    save(fig, **kwargs)

    return fig, axs
