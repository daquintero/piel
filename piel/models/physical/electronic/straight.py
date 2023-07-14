import hdl21 as h

__all__ = ["Straight", "StraightParameters"]


@h.paramclass
class StraightParameters:
    pass


@h.generator
def Straight(params: StraightParameters) -> h.Module:
    """
    Implements a `hdl21` taper resistor class.
    """
    return h.IdealResistor(r=1e3)
