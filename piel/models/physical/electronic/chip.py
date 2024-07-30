from ....types import ElectronicChip


def create_electronic_chip_component(
    name: str = None,
) -> ElectronicChip:
    """
    Create an electronic chip component.
    """
    if name is None:
        name = "electronic_chip"

    return ElectronicChip(
        name=name,
    )
