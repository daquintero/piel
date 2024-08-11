from ..types import DCSweepMeasurementData, DCSweepMeasurementDataCollection
from ...visual import create_plot_containers, save


def plot_dc_sweep(dc_sweep: DCSweepMeasurementData, **kwargs) -> tuple:
    """
    Plot a DC sweep measurement.

    Parameters
    ----------
    dc_sweep : DCMeasurementDataTypes
        The DC sweep measurement data to plot.
    """
    fig, axs = create_plot_containers(container_list=[dc_sweep.collection])

    axs[0].plot(
        # dc_sweep.inputs[0].signal.signal_instances[0].values
        dc_sweep.outputs[0].signal.signal_instances[0].values,
    )

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs


def plot_dc_sweeps(
    dc_sweep_collection: DCSweepMeasurementDataCollection, **kwargs
) -> tuple:
    fig, axs = create_plot_containers(
        container_list=dc_sweep_collection.collection, axes_structure="overlay"
    )

    for dc_sweep_i in dc_sweep_collection.collection:
        axs[0].plot(
            dc_sweep_i.inputs[0].signal.signal_instances[0].values,
            dc_sweep_i.outputs[0].signal.signal_instances[0].values,
        )

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs
