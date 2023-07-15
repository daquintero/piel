# from .taper import Taper
# from .straight import Straight

__all__ = ["get_default_models"]

import hdl21 as h


@h.paramclass
class MyParams:
    # Required
    width = h.Param(dtype=int, desc="Width. Required", default=10)
    # Optional - including a default value
    text = h.Param(dtype=str, desc="Optional string", default="My Favorite Module")


@h.generator
def MyFirstGenerator(params: MyParams) -> h.Module:
    # A very exciting first generator function
    m = h.Module()
    m.i = h.Input(width=params.width)
    return m


__default_models_dictionary__ = {
    "taper": MyFirstGenerator,
    "straight": MyFirstGenerator,
    "via_stack": MyFirstGenerator,
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
