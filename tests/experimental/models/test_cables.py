from piel.models.physical.electrical.cables.dc import (
    generic_banana,
)
from piel.models.physical.electrical.cables.rf import (
    rg164,
    generic_sma,
    cryo_cable,
)
from piel.types import CoaxialCable, DCCable


def test_generic_banana():
    cable = generic_banana(name="Banana Cable", length_m=1.5)

    assert isinstance(cable, DCCable)
    assert cable.name == "Banana Cable"
    assert len(cable.ports) == 2

    assert cable.ports[0].name == "IN"
    assert cable.ports[0].domain == "DC"
    assert cable.ports[0].connector == "Male"

    assert cable.ports[1].name == "OUT"
    assert cable.ports[1].domain == "DC"
    assert cable.ports[1].connector == "Male"


def test_rg164():
    cable = rg164(length_m=2.0)

    assert isinstance(cable, CoaxialCable)
    assert cable.name == "RG164"
    assert cable.geometry.length_m == 2.0
    assert cable.model == "RG164"
    assert len(cable.ports) == 2

    assert cable.ports[0].name == "IN"
    assert cable.ports[0].domain == "RF"
    assert cable.ports[0].connector == "SMA"

    assert cable.ports[1].name == "OUT"
    assert cable.ports[1].domain == "RF"
    assert cable.ports[1].connector == "SMA"


def test_generic_sma():
    cable = generic_sma(name="SMA Cable", length_m=3.0)

    assert isinstance(cable, CoaxialCable)
    assert cable.name == "SMA Cable"
    assert cable.geometry.length_m == 3.0
    assert len(cable.ports) == 2

    assert cable.ports[0].name == "IN"
    assert cable.ports[0].domain == "RF"
    assert cable.ports[0].connector == "SMA"

    assert cable.ports[1].name == "OUT"
    assert cable.ports[1].domain == "RF"
    assert cable.ports[1].connector == "SMA"


def test_cryo_cable():
    cable = cryo_cable(length_m=4.5)


# Add more tests as needed for additional behaviors, edge cases, and validation logic.
