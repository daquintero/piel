from piel.types.experimental.measurements.data.frequency import (
    VNASParameterMeasurementDataCollection,
)
from piel.types import MinimumMaximumType
from piel.visual import (
    create_plot_containers,
    save,
    create_axes_per_figure,
    create_axes_parameters_table_overlay,
    create_axes_parameters_tables_separate,
)

default_skrf_figure_kwargs = {
    "show_legend": False,
}


def plot_s_parameter_measurements_to_step_responses(
    data_collection: VNASParameterMeasurementDataCollection,
    parameters_list: list = None,
    network_port_index: int = 0,
    time_range_s: MinimumMaximumType = None,
    figure_kwargs: dict = None,
    **kwargs,
):
    """
    The goal of this function is that it iterates through a collection of s-parameter networks,
    generates the inverse-fourier-transform step responses and plots them as defined by the plotting infrastructure.
    Note that each step transmission depends on the corresponding input port for the subnetwork it is extracted from,
    as it is derived from the S11 or S22 based on the return loss, hence matching in a real transmission line network.

    It will plot the transformations on top of each other rather than sequentially.
    TODO explore the other caveats of performing transformations this way.
    TODO generalise this functionality for simulation-sparameter networks.
    """
    if figure_kwargs is None:
        figure_kwargs = dict()

    if parameters_list is None:
        parameters_list = range(len(data_collection.collection))

    fig, axs = create_plot_containers(
        container_list=data_collection.collection,
        axes_structure="overlay",
        **figure_kwargs,
    )

    fig.suptitle(f"{data_collection.name} Step Responses")

    i = 0
    for measurement_i in data_collection.collection:
        network = measurement_i.network
        if (network is None) or (network.number_of_ports == 0):
            print(
                f"Skipping network not found in the measurement data: {measurement_i}"
            )
        else:
            subnetwork = network.subnetwork(ports=[network_port_index])
            subnetwork_s11_time_i, subnetwork_s11_signal_i = subnetwork.step_response()

            if time_range_s is not None:
                axs[0].set_xlim(time_range_s[0], time_range_s[1])

            axs[0].plot(subnetwork_s11_time_i, subnetwork_s11_signal_i)

        i += 1

    if parameters_list is not None:
        if len(parameters_list) == len(data_collection.collection):
            # Create the labels accordingly
            try:
                create_axes_parameters_table_overlay(
                    fig=fig, axs=axs, parameters_list=parameters_list
                )
            except Exception as e:
                if "debug" in kwargs and kwargs.get("debug", False):
                    raise e
                pass

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs


def plot_s_parameter_real_and_imaginary(
    data_collection: VNASParameterMeasurementDataCollection,
    parameters_list: list = None,
    figure_kwargs: dict = None,
    s_plot_kwargs: dict = None,
    **kwargs,
) -> tuple:
    if figure_kwargs is None:
        figure_kwargs = dict()

    if s_plot_kwargs is None:
        s_plot_kwargs = default_skrf_figure_kwargs

    if parameters_list is None:
        parameters_list = range(len(data_collection.collection))

    fig, axs = create_plot_containers(
        container_list=data_collection.collection, **figure_kwargs
    )

    parameter_tables_list = list()

    i = 0
    for measurement_i in data_collection.collection:
        network = measurement_i.network
        if network is None:
            print(
                f"Skipping network not found in the measurement data: {measurement_i}"
            )
        else:
            network.plot_s_re(ax=axs[i], **s_plot_kwargs)
            # network.plot_s_im(ax=axs[1], **s_plot_configuration)
            parameter_tables_list.append(parameters_list[i])
            axs[i].set_title("Real S11")

        i += 1

    if parameters_list is not None:
        if len(parameters_list) == len(data_collection.collection):
            # Create the labels accordingly

            # TODO make tables list
            try:
                create_axes_parameters_tables_separate(
                    fig=fig, axs=axs, parameter_tables_list=parameter_tables_list
                )
            except Exception as e:
                if "debug" in kwargs and kwargs.get("debug", False):
                    raise e
                pass

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs


def plot_s_parameter_per_component(
    data_collection: VNASParameterMeasurementDataCollection,
    parameters_list: list = None,
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

    fig.suptitle(data_collection.name)

    # Create the plot titles
    axs[0, 0].set_title("S11")
    axs[0, 1].set_title("S12")
    axs[1, 0].set_title("S21")
    axs[1, 1].set_title("S22")

    # Create the ylabels
    axs[0, 0].set_ylabel(f"{s_parameter_plot}")
    axs[1, 0].set_ylabel(f"{s_parameter_plot}")
    axs[0, 0].set_ylabel(f"{s_parameter_plot}")
    axs[0, 0].set_ylabel(f"{s_parameter_plot}")

    # The goal of the following section is to plot the S-parameters in the corresponding axes of the figure

    i = 0
    for measurement_i in data_collection.collection:
        network = measurement_i.network

        # We want to extract the relevant parameter set from the network name here.

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
                        # s_plot_kwargs = {"label": parameters_list[i], **s_plot_kwargs}
                        getattr(network, s_parameter_plot)(
                            ax=axs[m, n], m=m, n=n, **s_plot_kwargs
                        )

                    except Exception as e:
                        print(f"Error plotting measurement: {measurement_i}")
                        print(f"Error plotting network: {network}")
                        print(f"Error plotting S-parameter: {e}")
                        if "debug" in kwargs and kwargs.get("debug", False):
                            raise e

        i += 1

    if parameters_list is not None:
        if len(parameters_list) == len(data_collection.collection):
            # Create the labels accordingly
            try:
                create_axes_parameters_table_overlay(
                    fig=fig, axs=axs, parameters_list=parameters_list
                )
            except Exception:
                pass

    # Save the figure if 'path' is provided in kwargs
    save(fig, **kwargs)

    return fig, axs
