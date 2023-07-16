"""
Hi Dan,

Hope you are well.
"""
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
    print("MyFirstGenerator")
    m = h.Module()
    m.i = h.Input(width=params.width)
    return m


@h.generator
def MySecondGenerator(params: MyParams) -> h.Module:
    # A very exciting first generator function
    m = h.Module()
    m.i = h.Input(width=params.width)
    m.a = MyFirstGenerator(width=params.width)(**{"i": m.i})
    # print(getattr(m, "i"))
    print("MyFirstGenerator ports")
    # m.i = m.a.i
    return m


a = MySecondGenerator(width=10)
print(a)
b = h.elaborate(a)
print(b)
# import sys

# from hdl21.elab.elaborators.conntypes import get_unconnected_instance_connections

# missing_conns = get_unconnected_instance_connections(b, inst=getattr(b, "a"))
# print(missing_conns)
# print(h.netlist(b, sys.stdout, fmt="spice"))


"""
RuntimeError: Elaboration Error at hierarchical path:
  Module        straight_heater_metal_s_b8a2a400s
  Instance      via_stack_1
Invalid connections `{'e1': Missing connection to Port `e1`, 'e2': Missing connection to Port `e2`, 'e4': Missing connection to Port `e4`}` on Instance `via_stack_1` in Module `straight_heater_metal_s_b8a2a400`

According to HDL21 currently, the existing internal ports do not solve external connectivity requirements in order to generate the netlist. This means that the declared ports at the top level must be connected. This means that we need a subroutine to fix this, and ideally it would be done by the generators but this is part of the composition. It would be good to have a helper function to extract this in terms of the composition. This may not be able to be done in the gdsfactory netlist, due to the nature of the connectivity. What would make sense is to expose them.

Part of the complexity is that the GeneratorCall class does not expose its ports until it is generated. So it can connect hypothetical ports, but it cannot expose the missing ports until after generation. We can use the unconencted ports warning though.
"""
