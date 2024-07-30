from ...types import CoaxialCable, CoaxialCableGeometryType, PhysicalPort, DCCable


def generic_banana(
    name: str,
    length_m: float,
    **kwargs,
) -> DCCable:
    ports = [
        PhysicalPort(
            name="IN",
            domain="DC",
            connector="Male",
        ),
        PhysicalPort(
            name="OUT",
            domain="DC",
            connector="Male",
        )
    ]

    return DCCable(
        name=name,
        ports=ports,
        **kwargs,
    )



def rg164(
    length_m: float,
    **kwargs,
) -> CoaxialCable:
    geometry = CoaxialCableGeometryType(
        length_m=length_m,
    )

    ports = [
        PhysicalPort(
            name="IN",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="OUT",
            domain="RF",
            connector="SMA",
        )
    ]

    return CoaxialCable(
        name="RG164",
        geometry=geometry,
        ports=ports,
        **kwargs,
    )


def generic_sma(
    name: str,
    length_m: float,
    **kwargs,
) -> CoaxialCable:
    geometry = CoaxialCableGeometryType(
        length_m=length_m,
    )

    ports = [
        PhysicalPort(
            name="IN",
            domain="RF",
            connector="SMA",
        ),
        PhysicalPort(
            name="OUT",
            domain="RF",
            connector="SMA",
        )
    ]

    return CoaxialCable(
        name=name,
        geometry=geometry,
        ports=ports,
        **kwargs,
    )


def cryo_cable(
    length_m: float,
) -> CoaxialCable:
    # TODO measure them
    geometry = CoaxialCableGeometryType(length_m=length_m)

    return CoaxialCable(geometry=geometry)
