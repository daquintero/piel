from piel.models.physical.electrical import create_two_port_oscilloscope, DPO73304
from piel.types.experimental import Oscilloscope
from piel.types import PhysicalPort


def test_create_two_port_oscilloscope():
    oscilloscope = create_two_port_oscilloscope()

    assert isinstance(oscilloscope, Oscilloscope)
    assert oscilloscope.name == "two_port_oscilloscope"
    assert len(oscilloscope.ports) == 2

    expected_ports = [
        {"name": "CH1", "domain": "RF", "connector": "SMA"},
        {"name": "CH2", "domain": "RF", "connector": "SMA"},
    ]

    for i, port in enumerate(oscilloscope.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


def test_create_two_port_oscilloscope_with_custom_name():
    oscilloscope = create_two_port_oscilloscope(name="CustomOscilloscope")

    assert oscilloscope.name == "CustomOscilloscope"


def test_dpo73304_initialization():
    oscilloscope = DPO73304()

    assert isinstance(oscilloscope, Oscilloscope)
    assert oscilloscope.name == "DPO73304"
    assert oscilloscope.manufacturer == "Tektronix"
    assert oscilloscope.model == "DPO73304"
    assert len(oscilloscope.ports) == 2

    expected_ports = [
        {"name": "CH1", "domain": "RF", "connector": "SMA"},
        {"name": "CH2", "domain": "RF", "connector": "SMA"},
    ]

    for i, port in enumerate(oscilloscope.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


def test_dpo73304_with_additional_kwargs():
    additional_kwargs = {"serial_number": "123456789"}
    oscilloscope = DPO73304(**additional_kwargs)

    assert oscilloscope.serial_number == "123456789"

    # Ensure no existing attributes are overridden
    assert oscilloscope.name == "DPO73304"
    assert oscilloscope.manufacturer == "Tektronix"
    assert oscilloscope.model == "DPO73304"
    assert len(oscilloscope.ports) == 2


# Add more tests as needed for additional behaviors, edge cases, and validation logic.
