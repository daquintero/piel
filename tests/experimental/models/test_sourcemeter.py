from piel.experimental.models import (
    create_dc_operating_point_configuration,
    create_dc_sweep_configuration,
    SMU2450,
)
from piel.experimental.types import (
    SourcemeterConfiguration,
    Sourcemeter,
)
from piel.types import PhysicalPort


def test_create_dc_operating_point_configuration():
    voltage_V = 5.0
    config = create_dc_operating_point_configuration(voltage_V=voltage_V)

    assert isinstance(config, SourcemeterConfiguration)
    assert config.voltage_set_V == voltage_V


def test_create_dc_sweep_configuration():
    voltage_range_V = (0.0, 10.0)
    config = create_dc_sweep_configuration(voltage_range_V=voltage_range_V)

    assert isinstance(config, SourcemeterConfiguration)
    assert config.voltage_range_V == voltage_range_V


def test_smu2450_initialization():
    sourcemeter = SMU2450()

    assert isinstance(sourcemeter, Sourcemeter)
    assert sourcemeter.name == "SMU2450"
    assert sourcemeter.manufacturer == "Keithley"
    assert sourcemeter.model == ""
    assert len(sourcemeter.ports) == 5

    expected_ports = [
        {"name": "FORCEHI", "domain": "DC", "connector": "Banana"},
        {"name": "FORCELO", "domain": "DC", "connector": "Banana"},
        {"name": "SENSEHI", "domain": "DC", "connector": "Banana"},
        {"name": "SENSELO", "domain": "DC", "connector": "Banana"},
        {"name": "MANIFOLDGND", "domain": "DC", "connector": "Banana"},
    ]

    for i, port in enumerate(sourcemeter.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


def test_smu2450_with_custom_name():
    sourcemeter = SMU2450(name="CustomSMU2450")

    assert sourcemeter.name == "CustomSMU2450"
    assert sourcemeter.manufacturer == "Keithley"
    assert sourcemeter.model == ""


# Add more tests as needed for additional behaviors, edge cases, and validation logic.
