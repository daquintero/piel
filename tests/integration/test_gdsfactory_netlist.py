import gdsfactory as gf
import functools
import sax

# Defines the resistance parameters
our_resistor = functools.partial(gf.components.straight_heater_metal, ohms_per_square=2)

our_resistive_heated_waveguide = functools.partial(
    gf.components.straight_heater_metal, ohms_per_square=2
)

# our_resistive_mzi_2x2_2x2_phase_shifter = mzi2x2_2x2_phase_shifter(
#     straight_x_top=our_resistive_heated_waveguide,
# )

our_electrical_gdsfactory_netlist = our_resistive_heated_waveguide().get_netlist(
    exclude_port_types="optical"
)

"""One of the main complexities of integrating the SPICE netlist is determining the exact instance connectivity. One
of the main complexities is that in `GDSFactory` many of these electrical components are provided in the form of
cross-sectional paths. This means that we need to find a way to extract connectivity that can be translated from
these fundamental structures.

The first step is that all `GDSFactory` components need to have electrical pins attached to them too, or they can not
be extracted accordingly. It would be ideal for this to be performed as part of a subroutine in the layout generation
scheme, like for the photonic components.

This means we have to add ports onto the cross-sections for particular instantiations of these sections.
 """
# straight().show()
# print(straight().get_netlist())
# print(sax.get_required_circuit_models(straight().get_netlist()))
# We can see the opticsal ports in layers 1/10

our_resistive_heated_waveguide().show()
# print(our_resistive_heated_waveguide())
print(our_electrical_gdsfactory_netlist["instances"].keys())
print(sax.get_required_circuit_models(our_electrical_gdsfactory_netlist))
# # ['via_stack'] # Currently this is being returned, but this is not an accurate circuit model.

# We can only see optical ports, and one set of electrical ports, but this is not the full connectivity.
# This means that we need to add the ports from the path.

"""See this basic example in the `gdsfactory` `straight` definition. This means we need to do this for all cross
sections however they are implemented. All instantiations of the cross-section use this straight component definition
so we need to include in some form electrical connectivity accordingly.

    p = gf.path.straight(length=length, npoints=npoints)
    x = gf.get_cross_section(cross_section, **kwargs)

    c = Component()
    path = p.extrude(x)
    ref = c << path
    c.add_ports(ref.ports)

Note that when you extract the netlist from the straight() component, it is a fundamental component so that you cannot create a circuit out of it.
{'connections': {}, 'instances': {}, 'placements': {}, 'ports': {}, 'name': 'straight'}

We can test with an electrical port:
"""
# straight(cross_section="metal1").show()
# print(straight(cross_section="metal1").get_netlist())
"""This does not add any pins to the electrical component. Note that we need to have a function to add them in order
to extract our electrical netlist connectivity. It would be reasonable, given that the electrical connectivity can be
fanned out, that the whole metal path has pins since geometrical connectivity can be performed from all directions
without DC loss. Now, on the RF side this is a different game, but for the sake of simple connectivity this can be
done. Note that we still have to apply our device parameters in between. The problem becomes defining an input or
output, because there is none, there is only both for fundamental physical connectivity.

Let's first explore how the standard implementation fares. In order to extract physical electrical connectivity,
then we need to make sure the pin connectivity is included in between metal layers, and that VIAs allow for
connectivity between layers. This needs to conform to the standard in tools such as Cadence. In this tool,
the pin process is a different purpose than the drawing layer. I can see in the generic PDK the WG_PIN layer it was already set to purpose 10 so maybe stick with it for the standard. """
# print(gdsfactory.pdk.get_layer_stack()["core"])

# print(sax.get_required_circuit_models(straight(cross_section="metal1").get_netlist()))


@gf.cell
def connected_metal():
    test = gf.Component()
    a = test << gf.components.straight(cross_section="metal1")
    test.add_ports(a.ports)
    return test


# connected_metal().show()
# print(connected_metal().get_netlist())
# print(sax.get_required_circuit_models(connected_metal().get_netlist()))
"""
This returns this:
{'connections': {}, 'instances': {'straight_1': {'component': 'straight', 'info': {'length': 10.0, 'width': 10.0, 'cross_section': 'metal1', 'settings':
 {'width': 10.0, 'offset': 0, 'layer': 'M1', 'width_wide': None, 'auto_widen': False, 'auto_widen_minimum_length': 200.0, 'taper_length': 10.0, 'radius'
: None, 'sections': None, 'port_names': ['e1', 'e2'], 'port_types': ['electrical', 'electrical'], 'gap': 5, 'min_length': 5, 'start_straight_length': 0.
01, 'end_straight_length': 0.01, 'snap_to_grid': None, 'bbox_layers': None, 'bbox_offsets': None, 'cladding_layers': None, 'cladding_offsets': None, 'cl
adding_simplify': None, 'info': None, 'decorator': None, 'add_pins': {'function': 'add_pins', 'settings': {'function': {'function': 'add_pin_rectangle_i
nside', 'settings': {'pin_length': 0.001, 'layer_label': None}}, 'layer': 'M1_PIN'}}, 'add_bbox': None, 'mirror': False, 'name': None}, 'function_name':
 'cross_section'}, 'settings': {'cross_section': 'metal1'}}}, 'placements': {'straight_1': {'x': 0.0, 'y': 0.0, 'rotation': 0, 'mirror': 0}}, 'ports': {
'e1': 'straight_1,e1', 'e2': 'straight_1,e2'}, 'name': 'connected_metal'}
['straight']

Now we're talking business as we can create models out of this.
"""


"""
So we need to add pins at different stages including optical and electronic.

Note that part of the complexity is that the heater connectivity doesn't make sense as it is right now.
"""
