import matplotlib.pyplot as plt
from ..types.measurements.data.frequency import VNASParameterMeasurementDataCollection
from ...types import MinimumMaximumType
from ...visual import create_plot_containers, save


def plot_s_parameter_measurements_to_step_responses(
    measurements: VNASParameterMeasurementDataCollection,
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

    fig, axs = create_plot_containers(container_list=measurements, **figure_kwargs)

    i = 0
    for measurement_i in measurements:
        subnetwork = measurement_i.network.subnetwork(ports=[network_port_index])
        subnetwork_s11_time_i, subnetwork_s11_signal_i = subnetwork.step_response()
        axs[i].plot(subnetwork_s11_time_i, subnetwork_s11_signal_i)

        if time_range_s is not None:
            axs[i].set_xlim(time_range_s[0], time_range_s[1])

        i += 1

    if kwargs["path"] is not None:
        save(fig, **kwargs)

    return fig, axs


def plot_s_parameter_real_and_imaginary(
    measurements: VNASParameterMeasurementDataCollection,
    figure_kwargs: dict = None,
    s_plot_kwargs: dict = None,
    **kwargs,
) -> tuple:
    if figure_kwargs is None:
        figure_kwargs = dict()

    if s_plot_kwargs is None:
        s_plot_kwargs = dict()

    fig, axs = create_plot_containers(container_list=measurements, **figure_kwargs)

    i = 0
    for measurement_i in measurements:
        network = measurement_i.network
        network.plot_s_re(ax=axs[i], **s_plot_kwargs)
        # network.plot_s_im(ax=axs[1], **s_plot_configuration)
        axs[i].set_title("Real S11")
        i += 1

    plt.tight_layout()

    if kwargs["path"]:
        save(fig, **kwargs)

    return fig, axs
