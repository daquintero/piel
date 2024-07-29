from ...types import PhysicalPort
from ..types import Sourcemeter


def SMU2450(**kwargs) -> Sourcemeter:
    ports = [
        PhysicalPort(
            name="FORCEHI",
            domain="DC",
            connector="Banana",
        ),
        PhysicalPort(
            name="FORCELO",
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

    return Sourcemeter(name="SMU2450", manufacturer="Keithley", ports=ports, **kwargs)
