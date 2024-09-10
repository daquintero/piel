from piel.types import PhysicalPort
from piel.types.experimental import Multimeter


def DMM6500(**kwargs) -> Multimeter:
    ports = [
        PhysicalPort(
            name="INPUTHI",
            domain="DC",
            connector="Banana",
        ),
        PhysicalPort(
            name="INPUTLO",
            domain="DC",
            connector="Banana",
        ),
        PhysicalPort(
            name="SENSEHI",
            domain="DC",
            connector="Banana",
        ),
        PhysicalPort(
            name="SENSELO",
            domain="DC",
            connector="Banana",
        ),
        PhysicalPort(
            name="MANIFOLDGND",
            domain="DC",
            connector="Banana",
        ),
    ]

    return Multimeter(
        name="DMM6500", ports=ports, manufacturer="Keithley", model="DMM6500", **kwargs
    )
