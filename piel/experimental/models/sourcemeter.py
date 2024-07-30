from typing import Optional
from ...types import PhysicalPort
from ..types import Sourcemeter


def SMU2450(name: Optional[str] = None,
            **kwargs) -> Sourcemeter:
    
    if name is None:
        name = "SMU2450"

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

    return Sourcemeter(name=name, manufacturer="Keithley", ports=ports, **kwargs)
