import matplotlib.pyplot as plt
from piel.experimental.types.measurements.data.frequency import (
    VNASParameterMeasurementDataCollection,
)
from piel.types import MinimumMaximumType
from piel.visual import create_plot_containers, save, create_axes_per_figure

default_skrf_figure_kwargs = {
    "show_legend": False,
}


def plot_s_parameter_measurements_to_step_responses(
    data_collection: VNASParameterMeasurementDataCollection,
    network_port_index: int = 0,
    time_range_s: MinimumMaximumType = None,
    figure_kwargs: dict = None,
    **kwargs,
):
    """
    The goal of this function is that it iterates through a collection of s-parameter networks,
    generates the inverse-fourier-transform step responses and plots them as defined by the plotting infrastructure.
    Note that each step response depends on the corresponding input port for the subnetwork it is extracted from,
    as it is derived from the S11 or S22 based on the return loss, hence matching in a real transmission line network.

    TODO explore the other caveats of performing transformations this way.
    TODO generalise this functionality for simulation-sparameter networks.
    """
    if figure_kwargs is None:
        figure_kwargs = dict()

    fig, axs = create_plot_containers(
        container_list=data_collection.collection, **figure_kwargs
    )

    i = 0
    for measurement_i in data_collection.collection:
        subnetwork = measurement_i.network.subnetwork(ports=[network_port_index])
        subnetwork_s11_time_i, subnetwork_s11_signal_i = subnetwork.step_response()
        axs[i].plot(subnetwork_s11_time_i, subnetwork_s11_signal_i)

        if time_range_s is not None:
            axs[i].set_xlim(time_range_s[0], time_range_s[1])

        i += 1

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs


def plot_s_parameter_real_and_imaginary(
    data_collection: VNASParameterMeasurementDataCollection,
    figure_kwargs: dict = None,
    s_plot_kwargs: dict = None,
    **kwargs,
) -> tuple:
    if figure_kwargs is None:
        figure_kwargs = dict()

    if s_plot_kwargs is None:
        s_plot_kwargs = default_skrf_figure_kwargs

    fig, axs = create_plot_containers(
        container_list=data_collection.collection, **figure_kwargs
    )

    i = 0
    for measurement_i in data_collection.collection:
        network = measurement_i.network
        network.plot_s_re(ax=axs[i], **s_plot_kwargs)
        # network.plot_s_im(ax=axs[1], **s_plot_configuration)
        axs[i].set_title("Real S11")
        i += 1

    plt.tight_layout()

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs


def plot_s_parameter_per_component(
    data_collection: VNASParameterMeasurementDataCollection,
    s_parameter_plot: str = "plot_s_db",
    figure_kwargs: dict = None,
    s_plot_kwargs: dict = None,
    **kwargs,
) -> tuple:
    """
    A set of two-port s-parameter measurements can have four different s-parameters, S11, S12, S21, S22.
    If we are wanting to visualize them under different operating conditions, it might be desired to create a
    separate plot for each of the s-parameters. This function is designed to do that. It assumes at least two `VNASParameterMeasurementData` are provided.

    Since a VNASParameterMeasurementDataCollection is a collection of VNASParameterMeasurementData,
     we can iterate through the collection and plot the S-parameters in each of the individual 4 set of plots.
    """

    if figure_kwargs is None:
        figure_kwargs = dict()

    if s_plot_kwargs is None:
        s_plot_kwargs = default_skrf_figure_kwargs

    # We generate four separate plots for each of the two-port S-parameters
    fig, axs = create_axes_per_figure(rows=2, columns=2, **figure_kwargs)

    # axs[i].set_title("Real S11")
    # The goal of the following section is to plot the S-parameters in the corresponding axes of the figure

    for measurement_i in data_collection.collection:
        network = measurement_i.network
        if network is None:
            print(
                f"Skipping network not found in the measurement data: {measurement_i}"
            )
        else:
            # Iterate through the s-parameters
            for m in range(2):
                for n in range(2):
                    # Create corresponding plot
                    try:
                        getattr(network, s_parameter_plot)(
                            ax=axs[m, n], m=m, n=n, **s_plot_kwargs
                        )
                    except Exception as e:
                        print(f"Error plotting measurement: {measurement_i}")
                        print(f"Error plotting network: {network}")
                        print(f"Error plotting S-parameter: {e}")

    plt.tight_layout()

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs
