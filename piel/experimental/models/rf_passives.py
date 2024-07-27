from ...types import PhysicalPort, PowerSplitter


def create_power_splitter_1to2():
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
        name="power_splitter_1to2",
        ports=ports,
    )
