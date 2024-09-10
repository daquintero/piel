from typing import Literal
from .photonic.straight_waveguide import (
    waveguide,
    lossless_straight,
    active_waveguide,
    ideal_lossless_active_waveguide,
)
from .photonic.mmi2x2 import mmi2x2_50_50
from .photonic.crossing_simple import crossing_simple

__all__ = [
    "get_default_models",
]

# Default model dictionary library that can be overwritten for specific modelling applications.
__default_models_dictionary__ = {
    "bend_euler": waveguide,
    "crossing45": crossing_simple,
    "mmi2x2": mmi2x2_50_50,
    "straight": waveguide,
    "straight_heater_metal_simple": active_waveguide,
    "straight_heater_metal_undercut": active_waveguide,
}

__default_quantum_models_dictionary__ = {
    "bend_euler": lossless_straight,
    "crossing45": crossing_simple,
    "mmi2x2": mmi2x2_50_50,
    "straight": lossless_straight,
    "straight_heater_metal_simple": active_waveguide,
    "straight_heater_metal_undercut": active_waveguide,
}

__default_classical_optical_function_verification_dictionary = {
    "bend_euler": lossless_straight,
    "crossing45": crossing_simple,
    "mmi2x2": mmi2x2_50_50,
    "straight": lossless_straight,
    "straight_heater_metal_simple": ideal_lossless_active_waveguide,
    "straight_heater_metal_undercut": ideal_lossless_active_waveguide,
}


def get_default_models(
    custom_defaults: dict | None = None,
    type: Literal["classical", "quantum", "optical_logic_verification"] = "classical",
) -> dict:
    """
    Returns the default measurement dictionary.

    Args:
        custom_defaults (dict): Custom defaults dictionary.
        type (Literal["default", "quantum"]): Type of default measurement dictionary to return.

    Returns:
        dict: Default measurement dictionary.
    """
    if custom_defaults is not None:
        return custom_defaults
    else:
        if type == "classical":
            return __default_models_dictionary__
        elif type == "quantum":
            return __default_quantum_models_dictionary__  #
        elif type == "optical_logic_verification":
            return __default_classical_optical_function_verification_dictionary
        else:
            raise ValueError(f"Type {type} not recognised.")
