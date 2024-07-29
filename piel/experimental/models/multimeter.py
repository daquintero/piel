from ...types import PhysicalPort
from ..types import Multimeter


def DMM6500() -> Multimeter:
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
        name="DMM6500",
        ports=ports,
        manufacturer="Keithley",
    )
