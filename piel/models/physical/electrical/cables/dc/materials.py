from piel.types.electrical.cables import (
    DCCableMaterialSpecificationType,
)
from piel.types.materials import MaterialReferenceType


def define_dc_cable_materials(
    core_material: MaterialReferenceType,
) -> DCCableMaterialSpecificationType:
    """
    Define the materials of a coaxial cable.

    Args:
        core_material: The material of the core.

    Returns:
        DCCableMaterialSpecificationType: The material specification of the dc cable.
    """
    return DCCableMaterialSpecificationType(
        core=core_material,
    )
