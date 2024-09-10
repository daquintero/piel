from piel.types import CoaxialCable, CoaxialCableGeometryType, PhysicalPort


def rg164(
    length_m: float,
    name: str = "RG164",
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
        ),
    ]

    return CoaxialCable(
        name=name,
        geometry=geometry,
        ports=ports,
        model="RG164",
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
        ),
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
