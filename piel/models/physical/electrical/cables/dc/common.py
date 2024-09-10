from piel.types import PhysicalPort, DCCable


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
        ),
    ]

    return DCCable(
        name=name,
        ports=ports,
        **kwargs,
    )
