from typing import Literal
from piel.models.physical.geometry import (
    calculate_cross_sectional_area_m2,
    awg_to_cross_sectional_area_m2,
)
from piel.types.electrical.cables import (
    DCCableGeometryType,
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
