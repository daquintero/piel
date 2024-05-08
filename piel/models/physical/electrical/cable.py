from typing import Optional, Literal

from ..geometry import calculate_cross_sectional_area_m2, awg_to_cross_sectional_area_m2
from ..thermal import heat_transfer_1d_W
from ....materials.thermal_conductivity.types import MaterialReferenceType
from piel.materials.thermal_conductivity.utils import get_thermal_conductivity_fit
from .types import *
from ..types import TemperatureRangeTypes


def calculate_coaxial_cable_geometry(
    length_m: float = 1,
    sheath_top_diameter_m: float = 1.651e-3,
    sheath_bottom_diameter_m: float = 1.468e-3,
    core_diameter_dimension: Literal["awg", "metric"] = "metric",
    core_diameter_awg: Optional[float] = None,
    core_diameter_m: float = 2e-3,
    **kwargs
) -> CoaxialCableGeometryType:
    """
    Calculate the geometry of a coaxial cable. Defaults are based on the parameters of a TODO

    Args:
        length_m: Length of the cable in meters.
        sheath_top_diameter_m: Diameter of the top of the sheath in meters.
        sheath_bottom_diameter_m: Diameter of the bottom of the sheath in meters.
        core_diameter_dimension: Dimension of the core diameter.
        core_diameter_awg: Core diameter in AWG.
        core_diameter_m: Core diameter in meters.
        **kwargs:

    Returns:
        CoaxialCableGeometryType: The geometry of the coaxial cable.
    """

    if core_diameter_dimension == "awg":
        core_diameter_m = awg_to_cross_sectional_area_m2(core_diameter_awg)

    core_cross_sectional_area_m2 = calculate_cross_sectional_area_m2(
        diameter_m=core_diameter_m
    )
    sheath_cross_sectional_area_m2 = calculate_cross_sectional_area_m2(
        diameter_m=sheath_top_diameter_m
    ) - calculate_cross_sectional_area_m2(diameter_m=sheath_bottom_diameter_m)
    total_cross_sectional_area_m2 = (
        core_cross_sectional_area_m2 + sheath_cross_sectional_area_m2
    )  # TODO dielectric

    return CoaxialCableGeometryType(
        length_m=length_m,
        core_cross_sectional_area_m2=core_cross_sectional_area_m2,
        sheath_cross_sectional_area_m2=sheath_cross_sectional_area_m2,
        total_cross_sectional_area_m2=total_cross_sectional_area_m2,
        **kwargs,
    )


def define_coaxial_cable_materials(
    core_material: MaterialReferenceType,
    sheath_material: MaterialReferenceType,
    dielectric_material: MaterialReferenceType,
) -> CoaxialCableMaterialSpecificationType:
    """
    Define the materials of a coaxial cable.

    Args:
        core_material: The material of the core.
        sheath_material: The material of the sheath.
        dielectric_material: The material of the dielectric.

    Returns:
        CoaxialCableMaterialSpecificationType: The material specification of the coaxial cable.
    """
    return CoaxialCableMaterialSpecificationType(
        core=core_material, sheath=sheath_material, dielectric=dielectric_material
    )


def calculate_coaxial_cable_heat_transfer(
    temperature_range_K: TemperatureRangeTypes,
    geometry_class: Optional[CoaxialCableGeometryType],
    material_class: Optional[CoaxialCableMaterialSpecificationType],
    core_material: Optional[MaterialReferenceType] = None,
    sheath_material: Optional[MaterialReferenceType] = None,
    dielectric_material: Optional[MaterialReferenceType] = None,
) -> CoaxialCableHeatTransferType:
    """
    Calculate the heat transfer of a coaxial cable.

    Args:
        temperature_range_K: The temperature range in Kelvin.
        geometry_class: The geometry of the cable.
        material_class: The material of the cable.
        core_material: The material of the core.
        sheath_material: The material of the sheath.
        dielectric_material: The material of the dielectric.

    Returns:
        CoaxialCableHeatTransferType: The heat transfer of the cable.
    """
    if material_class is not None:
        provided_materials = material_class.supplied_parameters()
    elif material_class is None:
        material_class = CoaxialCableMaterialSpecificationType(
            core=core_material, sheath=sheath_material, dielectric=dielectric_material
        )
        provided_materials = material_class.supplied_parameters()
    else:
        raise ValueError("No material class or material parameters provided.")

    heat_transfer_parameters = dict()
    total_heat_transfer_W = 0
    for material_i in provided_materials:
        thermal_conductivity_fit_i = get_thermal_conductivity_fit(
            temperature_range_K=temperature_range_K,
            material=getattr(material_class, material_i),
        )
        # CURRENT TODO compute the thermal conductivity fit accordingly. Implement a material reference to thermal conductivtiy data mapping.

        heat_transfer_i = heat_transfer_1d_W(
            thermal_conductivity_fit=thermal_conductivity_fit_i,
            temperature_range_K=temperature_range_K,
            cross_sectional_area_m2=geometry_class.total_cross_sectional_area_m2,
            length_m=geometry_class.length_m,
        )
        heat_transfer_parameters[material_i] = heat_transfer_i
        total_heat_transfer_W += heat_transfer_i
    heat_transfer_parameters["total"] = total_heat_transfer_W

    return CoaxialCableHeatTransferType(
        **heat_transfer_parameters,
    )


def calculate_dc_cable_geometry(
    length_m: float = 1,
    core_diameter_dimension: Literal["awg", "metric"] = "metric",
    core_diameter_awg: Optional[float] = None,
    core_diameter_m: float = 2e-3,
    *args,
    **kwargs
) -> DCCableGeometryType:
    """
    Calculate the geometry of a DC cable. Defaults are based on the parameters of a TODO

    Args:
        length_m: Length of the cable in meters.
        core_diameter_dimension: Dimension of the core diameter.
        core_diameter_awg: Core diameter in AWG.
        core_diameter_m: Core diameter in meters.
        **kwargs:

    Returns:
        CoaxialCableGeometryType: The geometry of the coaxial cable.
    """

    if core_diameter_dimension == "awg":
        core_diameter_m = awg_to_cross_sectional_area_m2(core_diameter_awg)

    core_cross_sectional_area_m2 = calculate_cross_sectional_area_m2(
        diameter_m=core_diameter_m
    )
    total_cross_sectional_area_m2 = core_cross_sectional_area_m2

    return DCCableGeometryType(
        length_m=length_m,
        core_cross_sectional_area_m2=core_cross_sectional_area_m2,
        total_cross_sectional_area_m2=total_cross_sectional_area_m2,
        *args,
        **kwargs,
    )


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


def calculate_dc_cable_heat_transfer(
    temperature_range_K: TemperatureRangeTypes,
    geometry_class: Optional[DCCableGeometryType],
    material_class: Optional[DCCableMaterialSpecificationType],
    core_material: Optional[MaterialReferenceType] = None,
) -> DCCableHeatTransferType:
    """
    Calculate the heat transfer of a coaxial cable.

    Args:
        temperature_range_K: The temperature range in Kelvin.
        geometry_class: The geometry of the cable.
        material_class: The material of the cable.
        core_material: The material of the core.

    Returns:
        DCCableHeatTransferType: The heat transfer of the cable.
    """

    if material_class is not None:
        provided_materials = material_class.supplied_parameters()
    elif material_class is None:
        material_class = DCCableMaterialSpecificationType(
            core=core_material,
        )
        provided_materials = material_class.supplied_parameters()
    else:
        raise ValueError("No material class or material parameters provided.")

    heat_transfer_parameters = dict()
    total_heat_transfer_W = 0
    for material_i in provided_materials:
        thermal_conductivity_fit_i = get_thermal_conductivity_fit(
            temperature_range_K=temperature_range_K,
            material=getattr(material_class, material_i),
        )
        # CURRENT TODO compute the thermal conductivity fit accordingly. Implement a material reference to thermal conductivtiy data mapping.

        heat_transfer_i = heat_transfer_1d_W(
            thermal_conductivity_fit=thermal_conductivity_fit_i,
            temperature_range_K=temperature_range_K,
            cross_sectional_area_m2=geometry_class.total_cross_sectional_area_m2,
            length_m=geometry_class.length_m,
        )
        heat_transfer_parameters[material_i] = heat_transfer_i
        total_heat_transfer_W += heat_transfer_i
    heat_transfer_parameters["total"] = total_heat_transfer_W

    return DCCableHeatTransferType(
        **heat_transfer_parameters,
    )
