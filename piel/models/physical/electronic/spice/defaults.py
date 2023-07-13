# from .capacitor import basic_capacitor
from .resistor import basic_resistor

__default_models_dictionary__ = {
    "taper": basic_resistor,
    "straight": basic_resistor,
    "via_stack": basic_resistor,
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
