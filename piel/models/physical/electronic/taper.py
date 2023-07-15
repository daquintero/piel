import hdl21 as h

__all__ = ["taper", "TaperParameters"]


@h.paramclass
class TaperParameters:
    pass


@h.generator
def taper(params: TaperParameters) -> h.Module:
    """
    Implements a `hdl21` taper resistor class. We need to include the mapping ports as we expect our gdsfactory component to be with the instance of the model.
    """

    @h.module
    class Taper:
        e1, e2 = h.Ports(2)
        r1 = h.IdealResistor(r=1e3)(p=e1, n=e2)

    return Taper
