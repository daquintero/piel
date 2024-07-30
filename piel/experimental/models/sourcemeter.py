from typing import Optional
from ...types import PhysicalPort
from ..types import Sourcemeter, SourcemeterConfiguration


def create_dc_operating_point_configuration(
    voltage_V: float,
) -> SourcemeterConfiguration:
    return SourcemeterConfiguration(voltage_set_V=voltage_V)


def create_dc_sweep_configuration(
    voltage_range_V: tuple[float, float],
) -> SourcemeterConfiguration:
    return SourcemeterConfiguration(voltage_range_V=voltage_range_V)


def SMU2450(name: Optional[str] = None, **kwargs) -> Sourcemeter:
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

    return Sourcemeter(
        name=name, manufacturer="Keithley", model="", ports=ports, **kwargs
    )
