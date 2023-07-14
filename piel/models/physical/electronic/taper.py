import hdl21 as h

__all__ = ["Taper", "TaperParameters"]


@h.paramclass
class TaperParameters:
    pass


@h.generator
def Taper(params: TaperParameters) -> h.Module:
    """
    Implements a `hdl21` taper resistor class.
    """
    h.a = h.IdealResistor(r=1e3)
    return h
