# TODO finish
from .photonic import straight_waveguide
from .photonic.mmi1x2 import mmi1x2_50_50
from .photonic.mmi2x2 import mmi2x2_50_50
from .photonic import coupler_simple

__all__ = [
    "get_all_models",
]

# Default model dictionary library that can be overwritten for specific modelling applications.
__default_models_dictionary__ = {
    "coupler": coupler_simple,
    "mmi1x2_50_50": mmi1x2_50_50,
    "mmi2x2_50_50": mmi2x2_50_50,
    "ideal_active_waveguide": straight_waveguide.active_waveguide,
    "simple_straight": straight_waveguide.simple_straight,
    "waveguide": straight_waveguide.waveguide,
}


def get_all_models(custom_library: dict | None = None) -> dict:
    """
    Returns the default measurement dictionary.

    Args:
        custom_library (dict): Custom defaults dictionary.

    Returns:
        dict: Default measurement dictionary.
    """
    if custom_library is not None:
        return custom_library
    else:
        return __default_models_dictionary__
