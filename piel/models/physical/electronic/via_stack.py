import hdl21 as h

__all__ = ["via_stack", "ViaStackParameters"]


@h.paramclass
class ViaStackParameters:
    pass


@h.generator
def via_stack(params: ViaStackParameters) -> h.Module:
    """
    Implements a `hdl21` taper resistor class. We need to include the mapping ports as we expect our gdsfactory component to be with the instance of the model.
    """

    @h.module
    class ViaStack:
        e1, e2, e3, e4, e5 = h.Ports(5)
        # All bottom ports connected together
        e1 = e2 = e3 = e4
        # Top port e5 is output of via
        h.r1 = h.IdealResistor(r=1e3)
        h.r1.p = e5
        h.r1.n = e1

    return ViaStack
