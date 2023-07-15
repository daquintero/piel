import hdl21 as h

__all__ = ["straight", "StraightParameters"]


@h.paramclass
class StraightParameters:
    """
    These are all the potential parametric configuration variables
    """

    pass


@h.generator
def straight(params: StraightParameters) -> h.Module:
    """
    Implements a `hdl21` taper resistor class.
    """

    @h.module
    class Straight:
        e1, e2 = h.Ports(2)
        h.r1 = h.IdealResistor(r=1e3)
        h.r1.p = e1
        h.r1.n = e2

    return Straight
