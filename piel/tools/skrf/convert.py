import numpy as np
from typing import List

# Import your custom Pydantic models
from piel.types.units import Hz, degree
from piel.types import NetworkTransmission, Phasor, PathTransmission


def convert_skrf_network_to_network_transmission(network) -> NetworkTransmission:
    """
    Converts a scikit-rf Network object to a NetworkTransmission object.

    Parameters:
    - network (rf.Network): The scikit-rf Network object to convert.

    Returns:
    - NetworkTransmission: The converted NetworkTransmission object.
    """

    # Extract frequency data
    frequencies = network.frequency.f  # Frequency values in Hz

    # Number of connection
    num_ports = network.number_of_ports

    # Initialize list to hold PathTransmission objects
    path_transmissions: List[PathTransmission] = []

    # Iterate over all S-parameters
    for i in range(num_ports):
        for j in range(num_ports):
            s_param = network.s[:, i, j]  # S-parameter Sij across all frequencies

            # Calculate magnitude in dBm and phase in degrees
            # Note: S-parameters are typically unitless; adjust units as needed
            magnitude = 20 * np.log10(np.abs(s_param))  # Convert to dB
            phase = np.angle(s_param, deg=True)  # Convert to degrees

            # Create Phasor object
            phasor = Phasor(
                magnitude=magnitude,
                phase=phase,
                frequency=frequencies,
            )

            # Define port mapping
            connection = (i + 1, j + 1)

            # Create PathTransmission object
            path_transmission = PathTransmission(
                connection=connection, transmission=phasor
            )

            path_transmissions.append(path_transmission)

    # Define input Phasor (for example, input port 1 across all frequencies)
    # This can be adjusted based on specific requirements
    input_s_param = network.s[:, 0, 0]  # Assuming port 1 is the input port

    input_magnitude = 20 * np.log10(np.abs(input_s_param))
    input_phase = np.angle(input_s_param, deg=True)

    input_phasor = Phasor(
        magnitude=input_magnitude,
        phase=input_phase,
        frequency=frequencies,
    )

    # Create NetworkTransmission object
    network_transmission = NetworkTransmission(
        input=input_phasor, network=path_transmissions
    )

    return network_transmission


# def convert_skrf_network_to_network_transmission(network, input_port: int = 0) -> NetworkTransmission:
#     """
#     Converts a scikit-rf Network object into a NetworkTransmission Pydantic model.
#
#     Args:
#         network (rf.Network): The scikit-rf Network object to convert.
#         input_port (int, optional): The port to consider as the input. Defaults to 0.
#
#     Returns:
#         NetworkTransmission: The converted NetworkTransmission object.
#     """
#     # TODO verify this is a scikit rf network
#
#     # Validate input_port
#     if input_port < 0 or input_port >= network.nports:
#         raise ValueError(f"input_port {input_port} is out of range for a network with {network.nports} connection.")
#
#     # Extract frequency data
#     frequencies = network.frequency.f  # Frequency in Hz
#     n_freq = len(frequencies)
#
#     # Extract S-parameters
#     s_params = network.s  # Shape: (n_freq, n_ports, n_ports)
#
#     # Define input Phasor using S-parameters from input_port to itself (e.g., S11)
#     # TODO check this
#     s_input = s_params[:, input_port, input_port]
#     input_magnitude = np.abs(s_input)
#     input_phase = np.angle(s_input, deg=True)
#
#     # Create Phasor for input
#     input_phasor = Phasor(
#         magnitude=input_magnitude,
#         phase=input_phase,
#         frequency=frequencies,
#         phase_unit=degree,
#         frequency_unit=Hz
#     )
#
#     # Define network transmissions
#     path_transmissions: List[PathTransmission] = []
#     n_ports = network.nports
#
#     for out_port in range(n_ports):
#         # Optionally skip the input port itself if not desired
#         # if out_port == input_port:
#         #     continue
#
#         # Define port mapping
#         connection = (input_port, out_port)
#
#         # Extract S-parameter from input_port to out_port
#         s_path = s_params[:, input_port, out_port]
#         trans_magnitude = np.abs(s_path)
#         trans_phase = np.angle(s_path, deg=True)
#
#         # Create Phasor for transmission
#         transmission_phasor = Phasor(
#             magnitude=trans_magnitude,
#             phase=trans_phase,
#             frequency=frequencies,
#             # magnitude_unit=Unit(name="linear"),  # Adjust unit if necessary
#             phase_unit=degree,
#             frequency_unit=Hz
#         )
#
#         # Create PathTransmission instance
#         path_transmission = PathTransmission(
#             connection=connection,
#             transmission=transmission_phasor
#         )
#
#         path_transmissions.append(path_transmission)
#
#     # Create NetworkTransmission instance
#     network_transmission = NetworkTransmission(
#         input=input_phasor,
#         network=path_transmissions
#     )
#
#     return network_transmission

# Example usage:
if __name__ == "__main__":
    import skrf as rf

    # Create a sample scikit-rf Network (for demonstration purposes)
    # In practice, you'd load a Network from a touchstone file or another source
    freq = rf.Frequency(1, 10, 100, "ghz")
    s = np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)
    sample_network = rf.Network(frequency=freq, s=s, name="Sample Network")

    # Convert to NetworkTransmission
    network_trans = convert_skrf_network_to_network_transmission(
        sample_network, input_port=0
    )

    # Access the converted data
    print(network_trans)
