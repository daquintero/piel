import hdl21 as h
from copy import copy

__all__ = ["taper"]


def taper(**kwargs) -> h.Module:
    """
    Implements a `hdl21` taper resistor class. We need to include the mapping ports as we expect our gdsfactory component to be with the instance of the model.
    """

    @h.module
    class Taper:
        name = kwargs["name"]
        e1, e2 = h.Ports(2)
        r1 = h.IdealResistor(r=1e3)(p=e1, n=e2)

    # TODO maybe chat about this with Dan Fritchman
    return copy(Taper)
