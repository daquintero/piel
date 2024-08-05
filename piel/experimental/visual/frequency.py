import matplotlib.pyplot as plt
from ..types.measurements.data.frequency import VNASParameterMeasurementDataCollection
from ...types import MinimumMaximumType


def plot_s_parameter_measurements_to_step_responses(
    measurements: VNASParameterMeasurementDataCollection,
    network_port_index: int = 0,
    time_range_s: MinimumMaximumType = None,
):
    """
    The goal of this function is that it iterates through a collection of s-parameter networks,
    generates the inverse-fourier-transform step responses and plots them as defined by the plotting infrastructure.
    Note that each step response depends on the corresponding input port for the subnetwork it is extracted from,
    as it is derived from the S11 or S22 based on the return loss, hence matching in a real transmission line network.

    TODO explore the other caveats of performing transformations this way.
    TODO generalise this functionality for simulation-sparameter networks.
    """
    for measurement_i in measurements:
        subnetwork = measurement_i.network.subnetwork(ports=[network_port_index])
        subnetwork_s11_time_i, subnetwork_s11_signal_i = subnetwork.step_response()
        plt.plot(subnetwork_s11_time_i, subnetwork_s11_signal_i)

    if time_range_s is not None:
        plt.xlim(time_range_s[0], time_range_s[1])
