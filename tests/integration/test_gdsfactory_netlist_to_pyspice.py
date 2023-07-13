import sax
import gdsfactory as gf


@gf.cell
def connected_metal():
    test = gf.Component()
    a = test << gf.components.straight(cross_section="metal1")
    b = test << gf.components.straight(cross_section="metal1")
    a.connect("e1", b.ports["e1"])
    test.add_port(port=a.ports["e2"], name="in")
    test.add_port(port=b.ports["e2"], name="out")
    return test


our_connected_metal = connected_metal()
# import json
# print(json.dumps(our_connected_metal.get_netlist()))
# print(our_connected_metal.get_netlist())

our_connected_metal.show()
print(sax.netlist(our_connected_metal.get_netlist()))
# print(dir(sax.netlist(connected_metal().get_netlist())))
"""
RecursiveNetlist(
    __root__={
        "top_level": Netlist(
            instances={
                "straight_1": Component(
                    component="straight", settings={"cross_section": "metal1"}
                ),
                "straight_2": Component(
                    component="straight", settings={"cross_section": "metal1"}
                ),
            },
            connections={"straight_1,e1": "straight_2,e1"},
            ports={"in": "straight_1,e2", "out": "straight_2,e2"},
            placements={
                "straight_1": Placement(
                    x="0.0",
                    y="0.0",
                    xmin=None,
                    ymin=None,
                    xmax=None,
                    ymax=None,
                    dx=0,
                    dy=0,
                    port=None,
                    rotation=180,
                    mirror=False,
                ),
                "straight_2": Placement(
                    x="0.0",
                    y="0.0",
                    xmin=None,
                    ymin=None,
                    xmax=None,
                    ymax=None,
                    dx=0,
                    dy=0,
                    port=None,
                    rotation=0,
                    mirror=False,
                ),
            },
        )
    }
)
"""
