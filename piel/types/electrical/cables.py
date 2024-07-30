from typing import Optional, Union
from piel.types.core import QuantityType
from piel.types.materials import MaterialReferenceType
from ..connectivity.physical import PhysicalConnection, PhysicalComponent, PhysicalPort


# TODO This shouldn't be a quantity
class CoaxialCableGeometryType(QuantityType):
    core_cross_sectional_area_m2: Optional[float] = None
    """
    The cross-sectional area of the core in meters squared.
    """

    length_m: float
    """
    The length of the cable in meters.
    """

    sheath_cross_sectional_area_m2: Optional[float] = None
    """
    The cross-sectional area of the sheath in meters squared.
    """

    total_cross_sectional_area_m2: Optional[float] = None
    """
    The total cross-sectional area of the cable in meters squared.
    """


class CoaxialCableHeatTransferType(QuantityType):
    """
    All units are in watts.
    """

    core: Optional[float] = None
    """
    The computed heat transfer in watts for the core of the cable.
    """

    sheath: Optional[float] = None
    """
    The computed heat transfer in watts for the sheath of the cable.
    """

    dielectric: Optional[float] = None
    """
    The computed heat transfer in watts for the dielectric of the cable.
    """

    total: float
    """
    The total computed heat transfer in watts for the cable.
    """

    units: str = "W"


class CoaxialCableMaterialSpecificationType(QuantityType):
    core: Optional[MaterialReferenceType] = None
    """
    The material of the core.
    """

    sheath: Optional[MaterialReferenceType] = None
    """
    The material of the sheath.
    """

    dielectric: Optional[MaterialReferenceType] = None
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

    core: Optional[float] = None
    """
    The computed heat transfer in watts for the core of the cable.
    """

    total: float
    """
    The total computed heat transfer in watts for the cable.
    """

    units: str = "W"


class DCCableMaterialSpecificationType(QuantityType):
    core: Optional[MaterialReferenceType] = None
    """
    The material of the core.
    """


class Cable(PhysicalComponent):
    pass


class DCCable(Cable):
    """
    A DC cable is a single core cable that is used to transmit direct current.
    """
    geometry: DCCableGeometryType
    heat_transfer: DCCableHeatTransferType
    material_specification: DCCableMaterialSpecificationType


class CoaxialCable(Cable):
    """
    A coaxial cable is a type of electrical cable that has an inner conductor surrounded by a tubular insulating layer,
    surrounded by a tubular conducting shield.
    """
    geometry: CoaxialCableGeometryType
    heat_transfer: CoaxialCableHeatTransferType
    material_specification: CoaxialCableMaterialSpecificationType


CableHeatTransferTypes = Union[CoaxialCableHeatTransferType, DCCableHeatTransferType]
CableGeometryTypes = Union[CoaxialCableGeometryType, DCCableGeometryType]
CableMaterialSpecificationTypes = Union[
    CoaxialCableMaterialSpecificationType, DCCableMaterialSpecificationType
]
