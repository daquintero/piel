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
