from typing import Optional, Literal, Callable
from piel.types.connectivity.physical import PhysicalConnection
from piel.types.connectivity.timing import TimeMetrics, TimeMetricsTypes
from ..geometry import calculate_cross_sectional_area_m2, awg_to_cross_sectional_area_m2
from ..thermal import heat_transfer_1d_W
from piel.materials.thermal_conductivity.utils import get_thermal_conductivity_fit
from piel.types.electrical.cables import (
    CoaxialCable,
    CoaxialCableGeometryType,
    CoaxialCableHeatTransferType,
    CoaxialCableMaterialSpecificationType,
    DCCableGeometryType,
    DCCableHeatTransferType,
    DCCableMaterialSpecificationType,
)
from piel.types.materials import MaterialReferenceType
from piel.types.physical import TemperatureRangeTypes
from piel.types import PhysicalPort
from piel.types import Connection


def calculate_coaxial_cable_geometry(
    length_m: float = 1,
    sheath_top_diameter_m: float = 1.651e-3,
    sheath_bottom_diameter_m: float = 1.468e-3,
    core_diameter_dimension: Literal["awg", "metric"] = "metric",
    core_diameter_awg: Optional[float] = None,
    core_diameter_m: float = 2e-3,
    **kwargs,
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
    temperature_range_K: TemperatureRangeTypes = [273, 293],
    geometry_class: CoaxialCableGeometryType = CoaxialCableGeometryType(),
    material_class: CoaxialCableMaterialSpecificationType | None = None,
    core_material: MaterialReferenceType | None = None,
    sheath_material: MaterialReferenceType | None = None,
    dielectric_material: MaterialReferenceType | None = None,
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
        # CURRENT TODO compute the thermal conductivity fit accordingly. Implement a material reference to thermal conductivtiy files mapping.

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
    core_diameter_awg: float = 0.0,
    core_diameter_m: float = 2e-3,
    *args,
    **kwargs,
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
    temperature_range_K: TemperatureRangeTypes = [273, 293],
    geometry_class: DCCableGeometryType = DCCableGeometryType(),
    material_class: DCCableMaterialSpecificationType
    | None = DCCableMaterialSpecificationType(),
    core_material: MaterialReferenceType = MaterialReferenceType(),
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
        # CURRENT TODO compute the thermal conductivity fit accordingly. Implement a material reference to thermal conductivtiy files mapping.

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


def create_coaxial_cable(
    material_specification_function: Callable[
        ..., CoaxialCableMaterialSpecificationType
    ]
    | None = None,
    timing_function: Callable[..., TimeMetricsTypes] | TimeMetricsTypes = TimeMetrics(),
    geometry_function: Callable[
        ..., CoaxialCableGeometryType
    ] = calculate_coaxial_cable_geometry,
    heat_transfer_function: Callable[
        ..., CoaxialCableHeatTransferType
    ] = calculate_coaxial_cable_heat_transfer,
    parameters: dict = {},
    **kwargs,
) -> CoaxialCable:
    """
    Creates a complete model of a CoaxialCable with relevant geometrical, frequency, timing, and heat transfer descriptions.

    This function operates on a collection of functions to create a comprehensive model of a `CoaxialCable`.
    Each function is parametrized through a `parameters` dictionary common to all defined internal functions,
    in order to compose each relevant model accordingly. This is decomposed internally within this method.

    Parameters:
    -----------
    material_specification_function : Callable[..., CoaxialCableMaterialSpecificationType] | None, optional
        A function that returns the material specification for the coaxial cable.
        This function should not take any arguments as it will be called without parameters.
        If None, no material specification will be set. Defaults to None.

    timing_function : Callable[..., TimeMetricsTypes] | TimeMetricsTypes, optional
        Either a function that calculates and returns the timing metrics for the coaxial cable,
        or a TimeMetricsTypes object directly.
        If a function, it will be called with the parameters from the `parameters` dict.
        Defaults to TimeMetrics().

    geometry_function : Callable[..., CoaxialCableGeometryType], optional
        A function that calculates and returns the geometry specification for the coaxial cable.
        Defaults to `calculate_coaxial_cable_geometry`.
        This function will be called with the parameters from the `parameters` dict.

    heat_transfer_function : Callable[..., CoaxialCableHeatTransferType], optional
        A function that calculates and returns the heat transfer characteristics of the coaxial cable.
        Defaults to `calculate_coaxial_cable_heat_transfer`.
        This function will be called with the parameters from the `parameters` dict.

    parameters : dict, optional
        A dictionary of parameters to be passed to the geometry, timing, and heat transfer functions.
        These parameters are used to customize the calculations for each aspect of the coaxial cable.
        Defaults to an empty dictionary.

    **kwargs :
        Additional keyword arguments to be passed to the CoaxialCable constructor.

    Returns:
    --------
    CoaxialCable
        A fully specified CoaxialCable object with all relevant properties set.

    Notes:
    ------
    - The function creates a Connection object with "in" and "out" PhysicalPorts, using the calculated time metrics.
    - A PhysicalConnection is created using the Connection object.
    - The CoaxialCable is constructed using the results from all calculation functions and the created PhysicalConnection.
    - If material_specification_function is None, no material specification will be set for the CoaxialCable.
    - The timing_function parameter can now be either a callable or a TimeMetricsTypes object directly.

    Example:
    --------
    >>> def material_spec():
    ...     return CoaxialCableMaterialSpecification(...)
    >>> def timing_calc(**params):
    ...     return TimeMetrics(...)
    >>> cable = create_coaxial_cable(
    ...     material_specification_function=material_spec,
    ...     timing_function=timing_calc,
    ...     parameters={'length': 10, 'diameter': 0.5},
    ...     name='My Coaxial Cable'
    ... )
    """
    heat_transfer = heat_transfer_function(**parameters)
    geometry = geometry_function(**parameters)

    if callable(timing_function):
        time_metrics = timing_function(**parameters)
    else:
        time_metrics = timing_function

    ports = [PhysicalPort(name="in"), PhysicalPort(name="out")]

    connection = Connection(
        ports=ports,
        time=time_metrics,
    )

    physical_connection = PhysicalConnection(connections=[connection])

    cable_kwargs = {
        "geometry": geometry,
        "heat_transfer": heat_transfer,
        "connections": [physical_connection],
        "ports": ports,
        **kwargs,
    }

    if material_specification_function is not None:
        cable_kwargs["material_specification"] = material_specification_function()

    return CoaxialCable(**cable_kwargs)
