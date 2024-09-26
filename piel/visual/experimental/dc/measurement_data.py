from piel.types import SignalDCCollection
from piel.visual import create_plot_containers, save


def plot_dc_sweep(dc_sweep: SignalDCCollection, **kwargs) -> tuple:
    """
    Plot a DC sweep measurement.

    Parameters
    ----------
    dc_sweep : DCMeasurementDataTypes
        The DC sweep measurement data to plot.
    """
    fig, axs = create_plot_containers(container_list=[dc_sweep.collection])

    axs[0].plot(
        # dc_sweep.inputs[0].signal.trace_list[0].values
        dc_sweep.outputs[0].signal.trace_list[0].values,
    )

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs
