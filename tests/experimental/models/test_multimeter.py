from piel.experimental.models import DMM6500
from piel.experimental.types import Multimeter
from piel.types import PhysicalPort


def test_dmm6500_initialization():
    multimeter = DMM6500()

    assert isinstance(multimeter, Multimeter)
    assert multimeter.name == "DMM6500"
    assert multimeter.manufacturer == "Keithley"
    assert multimeter.model == "DMM6500"
    assert len(multimeter.ports) == 5

    expected_ports = [
        {"name": "INPUTHI", "domain": "DC", "connector": "Banana"},
        {"name": "INPUTLO", "domain": "DC", "connector": "Banana"},
        {"name": "SENSEHI", "domain": "DC", "connector": "Banana"},
        {"name": "SENSELO", "domain": "DC", "connector": "Banana"},
        {"name": "MANIFOLDGND", "domain": "DC", "connector": "Banana"},
    ]

    for i, port in enumerate(multimeter.ports):
        assert isinstance(port, PhysicalPort)
        assert port.name == expected_ports[i]["name"]
        assert port.domain == expected_ports[i]["domain"]
        assert port.connector == expected_ports[i]["connector"]


def test_dmm6500_with_additional_kwargs():
    additional_kwargs = {"serial_number": "123456789"}
    multimeter = DMM6500(**additional_kwargs)

    assert multimeter.serial_number == "123456789"

    # Ensure no existing attributes are overridden
    assert multimeter.name == "DMM6500"
    assert multimeter.manufacturer == "Keithley"
    assert multimeter.model == "DMM6500"
    assert len(multimeter.ports) == 5


# Add more tests as needed for additional edge cases and behaviors.
