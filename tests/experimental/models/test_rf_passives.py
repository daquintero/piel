from piel.experimental.models import (
    create_power_splitter_1to2,
    create_bias_tee,
    create_attenuator,
    Picosecond5575A104,
)
from piel.types import PowerSplitter, BiasTee, PhysicalPort


def test_create_power_splitter_1to2():
    splitter = create_power_splitter_1to2()

    assert isinstance(splitter, PowerSplitter)
    assert splitter.name == "power_splitter_1to2"
    assert len(splitter.ports) == 3

    expected_ports = [
        {"name": "IN", "domain": "RF", "connector": "SMA"},
        {"name": "OUT1", "domain": "RF", "connector": "SMA"},
        {"name": "OUT2", "domain": "RF", "connector": "SMA"},
    ]

    for i, port in enumerate(splitter.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


def test_create_bias_tee():
    bias_tee = create_bias_tee()

    assert isinstance(bias_tee, BiasTee)
    assert bias_tee.name == "bias_tee"
    assert len(bias_tee.ports) == 3

    expected_ports = [
        {"name": "IN_RF", "domain": "RF", "connector": "SMA"},
        {"name": "IN_DC", "domain": "RF", "connector": "SMA"},
        {"name": "OUT", "domain": "RF", "connector": "SMA"},
    ]

    for i, port in enumerate(bias_tee.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


def test_create_attenuator():
    attenuator = create_attenuator()

    assert isinstance(attenuator, BiasTee)
    assert attenuator.name == "attenuator"
    assert len(attenuator.ports) == 2

    expected_ports = [
        {"name": "RF", "domain": "RF", "connector": "SMA"},
        {"name": "OUT", "domain": "RF", "connector": "SMA"},
    ]

    for i, port in enumerate(attenuator.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


def test_picosecond_5575a104():
    bias_tee = Picosecond5575A104()

    assert isinstance(bias_tee, BiasTee)
    assert bias_tee.name == "bias_tee"
    assert bias_tee.manufacturer == "Picosecond Pulse Labs"
    assert bias_tee.model == "5575A104"
    assert len(bias_tee.ports) == 3

    expected_ports = [
        {"name": "IN_RF", "domain": "RF", "connector": "SMA"},
        {"name": "IN_DC", "domain": "RF", "connector": "SMA"},
        {"name": "OUT", "domain": "RF", "connector": "SMA"},
    ]

    for i, port in enumerate(bias_tee.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


# Add more tests as needed for additional behaviors, edge cases, and validation logic.
