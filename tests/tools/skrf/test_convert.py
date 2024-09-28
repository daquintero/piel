import pytest
import numpy as np
import skrf as rf

# Import the function to be tested
from piel.tools.skrf import (
    convert_skrf_network_to_network_transmission,
)  # Replace 'your_module' with the actual module name


@pytest.fixture
def single_port_network():
    """
    Fixture to create a single-port scikit-rf Network object with known S-parameters.
    """
    freq = rf.Frequency(1, 2, 3, "ghz")  # 3 frequency points: 1 GHz, 1.5 GHz, 2 GHz
    s = np.array(
        [
            [[0.5 + 0.5j]],  # S11 at 1 GHz
            [[0.6 + 0.4j]],  # S11 at 1.5 GHz
            [[0.7 + 0.3j]],  # S11 at 2 GHz
        ]
    )
    network = rf.Network(frequency=freq, s=s, name="Single Port Network")
    return network


@pytest.fixture
def two_port_network():
    """
    Fixture to create a two-port scikit-rf Network object with known S-parameters.
    """
    freq = rf.Frequency(1, 3, 3, "ghz")  # 3 frequency points: 1 GHz, 2 GHz, 3 GHz
    s = np.array(
        [
            [[0.5 + 0.1j, 0.2 + 0.05j], [0.3 + 0.07j, 0.4 + 0.09j]],
            [[0.6 + 0.2j, 0.25 + 0.06j], [0.35 + 0.08j, 0.45 + 0.1j]],
            [[0.7 + 0.3j, 0.3 + 0.07j], [0.4 + 0.09j, 0.5 + 0.11j]],
        ]
    )
    network = rf.Network(frequency=freq, s=s, name="Two Port Network")
    return network


def test_single_port_conversion(single_port_network):
    """
    Test conversion of a single-port network.
    """
    network_trans = convert_skrf_network_to_network_transmission(single_port_network)

    # Assert input Phasor
    expected_magnitude = 20 * np.log10(np.abs(single_port_network.s[:, 0, 0]))
    expected_phase = np.angle(single_port_network.s[:, 0, 0], deg=True)

    assert network_trans.input.magnitude.tolist() == pytest.approx(
        expected_magnitude.tolist(), rel=1e-5
    )
    assert network_trans.input.phase.tolist() == pytest.approx(
        expected_phase.tolist(), rel=1e-5
    )
    assert (
        network_trans.input.frequency.tolist()
        == single_port_network.frequency.f.tolist()
    )

    # Assert network transmissions
    assert len(network_trans.network) == 1  # Only S11

    path = network_trans.network[0]
    assert path.ports == (1, 1)
    assert path.transmission.magnitude.tolist() == pytest.approx(
        expected_magnitude.tolist(), rel=1e-5
    )
    assert path.transmission.phase.tolist() == pytest.approx(
        expected_phase.tolist(), rel=1e-5
    )
    assert (
        path.transmission.frequency.tolist() == single_port_network.frequency.f.tolist()
    )


def test_two_port_conversion(two_port_network):
    """
    Test conversion of a two-port network.
    """
    network_trans = convert_skrf_network_to_network_transmission(two_port_network)

    # Assert input Phasor (S11)
    expected_magnitude = 20 * np.log10(np.abs(two_port_network.s[:, 0, 0]))
    expected_phase = np.angle(two_port_network.s[:, 0, 0], deg=True)

    assert network_trans.input.magnitude.tolist() == pytest.approx(
        expected_magnitude.tolist(), rel=1e-5
    )
    assert network_trans.input.phase.tolist() == pytest.approx(
        expected_phase.tolist(), rel=1e-5
    )
    assert (
        network_trans.input.frequency.tolist() == two_port_network.frequency.f.tolist()
    )

    # Assert network transmissions (S11, S12, S21, S22)
    assert len(network_trans.network) == 4

    # Define expected port mappings and their corresponding S-parameters
    expected_paths = {
        (1, 1): two_port_network.s[:, 0, 0],
        (1, 2): two_port_network.s[:, 0, 1],
        (2, 1): two_port_network.s[:, 1, 0],
        (2, 2): two_port_network.s[:, 1, 1],
    }

    for path in network_trans.network:
        ports = path.ports
        s_param = expected_paths[ports]
        expected_mag = 20 * np.log10(np.abs(s_param))
        expected_ph = np.angle(s_param, deg=True)

        assert path.transmission.magnitude.tolist() == pytest.approx(
            expected_mag.tolist(), rel=1e-5
        )
        assert path.transmission.phase.tolist() == pytest.approx(
            expected_ph.tolist(), rel=1e-5
        )
        assert (
            path.transmission.frequency.tolist()
            == two_port_network.frequency.f.tolist()
        )


def test_zero_s_parameters():
    """
    Test conversion when S-parameters are zero.
    """
    freq = rf.Frequency(1, 1, 1, "ghz")  # Single frequency point
    s = np.array([[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]]])
    network = rf.Network(frequency=freq, s=s, name="Zero S-Parameters Network")

    network_trans = convert_skrf_network_to_network_transmission(network)

    # Since log10(0) is undefined, np.log10 will return -inf. Handle accordingly.
    expected_magnitude = [-np.inf, -np.inf, -np.inf, -np.inf]
    expected_phase = [0.0, 0.0, 0.0, 0.0]  # Phase is 0 for zero magnitude

    # Input Phasor (S11)
    assert network_trans.input.magnitude[0] == pytest.approx(-np.inf)
    assert network_trans.input.phase[0] == pytest.approx(0.0)

    # Network transmissions
    assert len(network_trans.network) == 4
    for path in network_trans.network:
        assert path.transmission.magnitude[0] == pytest.approx(-np.inf)
        assert path.transmission.phase[0] == pytest.approx(0.0)


def test_frequency_assignment(two_port_network):
    """
    Test that the frequency data is correctly assigned in the NetworkTransmission object.
    """
    network_trans = convert_skrf_network_to_network_transmission(two_port_network)

    expected_frequencies = two_port_network.frequency.f.tolist()

    # Check input Phasor frequencies
    assert network_trans.input.frequency.tolist() == expected_frequencies

    # Check all network transmissions frequencies
    for path in network_trans.network:
        assert path.transmission.frequency.tolist() == expected_frequencies


def test_multiple_ports():
    """
    Test conversion of a network with more than two ports.
    """
    # Create a 3-port network
    freq = rf.Frequency(1, 3, 3, "ghz")  # 3 frequency points
    s = np.array(
        [
            [
                [0.5 + 0.1j, 0.2 + 0.05j, 0.1 + 0.02j],
                [0.3 + 0.07j, 0.4 + 0.09j, 0.15 + 0.03j],
                [0.1 + 0.02j, 0.15 + 0.03j, 0.6 + 0.1j],
            ],
            [
                [0.6 + 0.2j, 0.25 + 0.06j, 0.12 + 0.025j],
                [0.35 + 0.08j, 0.45 + 0.1j, 0.18 + 0.04j],
                [0.12 + 0.025j, 0.18 + 0.04j, 0.65 + 0.12j],
            ],
            [
                [0.7 + 0.3j, 0.3 + 0.07j, 0.14 + 0.03j],
                [0.4 + 0.09j, 0.5 + 0.11j, 0.21 + 0.05j],
                [0.14 + 0.03j, 0.21 + 0.05j, 0.7 + 0.14j],
            ],
        ]
    )
    network = rf.Network(frequency=freq, s=s, name="Three Port Network")

    network_trans = convert_skrf_network_to_network_transmission(network)

    # Assert input Phasor (S11)
    expected_magnitude = 20 * np.log10(np.abs(network.s[:, 0, 0]))
    expected_phase = np.angle(network.s[:, 0, 0], deg=True)

    assert network_trans.input.magnitude.tolist() == pytest.approx(
        expected_magnitude.tolist(), rel=1e-5
    )
    assert network_trans.input.phase.tolist() == pytest.approx(
        expected_phase.tolist(), rel=1e-5
    )
    assert network_trans.input.frequency.tolist() == network.frequency.f.tolist()

    # Assert network transmissions (9 paths)
    assert len(network_trans.network) == 9

    # Define expected port mappings and their corresponding S-parameters
    expected_paths = {}
    n_ports = 3
    for i in range(n_ports):
        for j in range(n_ports):
            expected_paths[(i + 1, j + 1)] = network.s[:, i, j]

    for path in network_trans.network:
        ports = path.ports
        s_param = expected_paths[ports]
        expected_mag = 20 * np.log10(np.abs(s_param))
        expected_ph = np.angle(s_param, deg=True)

        assert path.transmission.magnitude.tolist() == pytest.approx(
            expected_mag.tolist(), rel=1e-5
        )
        assert path.transmission.phase.tolist() == pytest.approx(
            expected_ph.tolist(), rel=1e-5
        )
        assert path.transmission.frequency.tolist() == network.frequency.f.tolist()


def test_non_standard_units(two_port_network):
    """
    Test conversion when frequency units are not standard (e.g., MHz instead of GHz).
    """
    # Modify frequency units to MHz
    network = two_port_network.copy()
    network.frequency.unit = "mhz"

    network_trans = convert_skrf_network_to_network_transmission(network)

    # Assert frequencies are correctly assigned in Hz (assuming conversion inside the function)
    # If the function expects Hz, but the network has MHz, this might need handling
    # Adjust the expected frequencies accordingly
    expected_frequencies = network.frequency.f.tolist()  # Assuming f is in Hz

    assert network_trans.input.frequency.tolist() == expected_frequencies
    for path in network_trans.network:
        assert path.transmission.frequency.tolist() == expected_frequencies
