from .photonic.straight_waveguide import waveguide
from .photonic.mmi2x2 import mmi2x2_50_50

__all__ = [
    "get_default_models",
]

# Default model dictionary library that can be overwritten for specific modelling applications.
__default_models_dictionary__ = {
    "bend_euler": waveguide,
    "mmi2x2": mmi2x2_50_50,
    "straight": waveguide,
}


def get_default_models(custom_defaults: dict | None = None) -> dict:
    """
    Returns the default models dictionary.

    Args:
        custom_defaults (dict): Custom defaults dictionary.

    Returns:
        dict: Default models dictionary.
    """
    if custom_defaults is not None:
        return custom_defaults
    else:
        return __default_models_dictionary__
