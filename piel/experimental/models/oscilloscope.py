from ...types import PhysicalPort
from ..types import Oscilloscope


def create_two_port_oscilloscope(
    name: str = "two_port_oscilloscope", **kwargs
) -> Oscilloscope:
    ports = [
        PhysicalPort(
            name="CH1",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="CH2",
            domain="RF",
            connector="SMA",
        ),
    ]

    return Oscilloscope(
        name=name,
        ports=ports,
    )


def DPO73304(name: str = "DPO73304", **kwargs) -> Oscilloscope:
    ports = [
        PhysicalPort(
            name="CH1",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="CH2",
            domain="RF",
            connector="SMA",
        ),
    ]

    return Oscilloscope(
        name=name, ports=ports, manufacturer="Tektronix", model="DPO73304", **kwargs
    )
