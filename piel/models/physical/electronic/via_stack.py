import hdl21 as h

__all__ = ["via_stack"]


def via_stack(**kwargs) -> h.Module:
    """
    Implements a `hdl21` taper resistor class. We need to include the mapping ports as we expect our gdsfactory component to be with the instance of the model.
    """

    @h.module
    class ViaStack:
        # e1, e2, e3, e4, e5 = h.Ports(5)
        e2, e3 = h.Ports(2)
        # TODO PR this  to hdl21
        # All bottom ports connected together
        # e1 = e2 = e3 = e4
        # Top port e5 is output of via
        r1 = h.IdealResistor(r=1e3)(p=e2, n=e3)

    return ViaStack
