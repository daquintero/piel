import hdl21 as h
from copy import copy

__all__ = ["straight"]


def straight(**kwargs) -> h.Module:
    """
    Implements a `hdl21` taper resistor class.
    """

    @h.module
    class Straight:
        e1, e2 = h.Ports(2)
        r1 = h.IdealResistor(r=1e3)(p=e1, n=e2)

    return copy(Straight)
