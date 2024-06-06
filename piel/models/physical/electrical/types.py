from typing import Optional, Union
from ....materials.thermal_conductivity.types import MaterialReferenceType
from ....types import QuantityType

__all__ = [
    "CoaxialCableGeometryType",
    "CoaxialCableHeatTransferType",
    "CoaxialCableMaterialSpecificationType",
    "DCCableGeometryType",
    "DCCableHeatTransferType",
    "DCCableMaterialSpecificationType",
    "CableHeatTransferTypes",
    "CableGeometryTypes",
    "CableMaterialSpecificationTypes",
]


# TODO This shouldn't be a quantity
class CoaxialCableGeometryType(QuantityType):
    core_cross_sectional_area_m2: Optional[float]
    """
    The cross-sectional area of the core in meters squared.
    """

    length_m: float
    """
    The length of the cable in meters.
    """

    sheath_cross_sectional_area_m2: Optional[float]
    """
    The cross-sectional area of the sheath in meters squared.
    """

    total_cross_sectional_area_m2: Optional[float]
    """
    The total cross-sectional area of the cable in meters squared.
    """


class CoaxialCableHeatTransferType(QuantityType):
    """
    All units are in watts.
    """

    core: Optional[float]
    """
    The computed heat transfer in watts for the core of the cable.
    """

    sheath: Optional[float]
    """
    The computed heat transfer in watts for the sheath of the cable.
    """

    dielectric: Optional[float]
    """
    The computed heat transfer in watts for the dielectric of the cable.
    """

    total: float
    """
    The total computed heat transfer in watts for the cable.
    """

    units: str = "W"


class CoaxialCableMaterialSpecificationType(QuantityType):
    core: Optional[MaterialReferenceType]
    """
    The material of the core.
    """

    sheath: Optional[MaterialReferenceType]
    """
    The material of the sheath.
    """

    dielectric: Optional[MaterialReferenceType]
    """
    The material of the dielectric.
    """


class DCCableGeometryType(QuantityType):
    core_cross_sectional_area_m2: float
    """
    The cross-sectional area of the core in meters squared.
    """

    length_m: float
    """
    The length of the cable in meters.
    """

    total_cross_sectional_area_m2: float
    """
    The total cross-sectional area of the cable in meters squared.
    """


class DCCableHeatTransferType(QuantityType):
    """
    All units are in watts.
    """

    core: Optional[float]
    """
    The computed heat transfer in watts for the core of the cable.
    """

    total: float
    """
    The total computed heat transfer in watts for the cable.
    """

    units: str = "W"


class DCCableMaterialSpecificationType(QuantityType):
    core: Optional[MaterialReferenceType]
    """
    The material of the core.
    """


CableHeatTransferTypes = Union[CoaxialCableHeatTransferType, DCCableHeatTransferType]
CableGeometryTypes = Union[CoaxialCableGeometryType, DCCableGeometryType]
CableMaterialSpecificationTypes = Union[
    CoaxialCableMaterialSpecificationType, DCCableMaterialSpecificationType
]
