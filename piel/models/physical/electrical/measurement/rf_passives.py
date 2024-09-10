from typing import Optional
from functools import partial
from piel.types import PhysicalPort, PowerSplitter, BiasTee


def create_power_splitter_1to2(name: Optional[str] = None):
    if name is None:
        name = "power_splitter_1to2"

    ports = [
        PhysicalPort(
            name="IN",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="OUT1",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="OUT2",
            domain="RF",
            connector="SMA",
        ),
    ]

    return PowerSplitter(
        name=name,
        ports=ports,
    )


def create_bias_tee(name: Optional[str] = None, **kwargs):
    if name is None:
        name = "bias_tee"

    ports = [
        PhysicalPort(
            name="IN_RF",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="IN_DC",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="OUT",
            domain="RF",
            connector="SMA",
        ),
    ]

    return BiasTee(name=name, ports=ports, **kwargs)


def create_attenuator(name: Optional[str] = None, **kwargs):
    if name is None:
        name = "attenuator"

    ports = [
        PhysicalPort(
            name="RF",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="OUT",
            domain="RF",
            connector="SMA",
        ),
    ]

    return BiasTee(
        name=name,
        ports=ports,
    )


Picosecond5575A104 = partial(
    create_bias_tee, manufacturer="Picosecond Pulse Labs", model="5575A104"
)
