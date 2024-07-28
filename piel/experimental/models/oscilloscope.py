from ...types import PhysicalPort
from ..types import Oscilloscope


def create_two_port_oscilloscope() -> Oscilloscope:
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
        name="two_port_oscilloscope",
        ports=ports,
    )


def DPO73304():
    pass
