from ...types import CoaxialCable, CoaxialCableGeometryType


def rg164(
    length_m: float,
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
    )


def cryo_cable(
    length_m: float,
) -> CoaxialCable:
    # TODO measure them
    geometry = CoaxialCableGeometryType(length_m=length_m)

    return CoaxialCable(geometry=geometry)
