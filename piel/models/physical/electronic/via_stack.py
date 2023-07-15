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
        """
        This is unrealistic but need to PR to hdl21 for validitiy
        """

        e1, e2, e3, e4 = h.Ports(4)
        # TODO PR this valid to hdl21
        # All bottom ports connected together
        # e1 = e2 = e3 = e4
        # Top port e5 is output of via
        r1 = h.IdealResistor(r=1e3)(p=e1, n=e2)
        r2 = h.IdealResistor(r=1e3)(p=e3, n=e4)
        r1.p = r2.p

    return ViaStack
