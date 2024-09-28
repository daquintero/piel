from typing import Callable
from piel.types.connectivity.physical import PhysicalConnection
from piel.types.connectivity.timing import TimeMetric, TimeMetricsTypes
from piel.types.electrical.cables import (
    CoaxialCable,
    CoaxialCableGeometryType,
    CoaxialCableHeatTransferType,
    CoaxialCableMaterialSpecificationType,
)
from piel.types import PhysicalPort
from piel.types import Connection
from .geometry import calculate_coaxial_cable_geometry
from .thermal import calculate_coaxial_cable_heat_transfer


def create_coaxial_cable(
    material_specification_function: Callable[
        ..., CoaxialCableMaterialSpecificationType
    ]
    | None = None,
    timing_function: Callable[..., TimeMetricsTypes] | TimeMetricsTypes = TimeMetric(),
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
    ...     return TimeMetric(...)
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
