from .taper import taper
from .straight import straight
from .via_stack import via_stack

__all__ = ["get_default_models"]


__default_models_dictionary__ = {
    "taper": taper,
    "straight": straight,
    "via_stack": via_stack,
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
